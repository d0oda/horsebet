"""
UmaEdge — Backfill Missing Sectional Times (last_3f_secs / first_3f_secs).

Re-scrapes finished races that have result rows but are missing last_3f_secs.
Updates existing rows in the `results` table with sectional data.

Usage:
    python -m scraper.backfill_sectionals                    # backfill all missing
    python -m scraper.backfill_sectionals --dry-run          # preview only
    python -m scraper.backfill_sectionals --limit 50         # limit races
    python -m scraper.backfill_sectionals --workers 3        # concurrency
    python -m scraper.backfill_sectionals --year 2021        # specific year only
"""

import argparse
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from sqlalchemy import text

from scraper.db import get_session
from scraper.netkeiba import (
    RESULT_URL,
    _fetch,
    parse_race_page,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_sectionals")


def get_races_missing_sectionals(year: Optional[int] = None, limit: Optional[int] = None) -> list[dict]:
    """Find races that have result rows with missing last_3f_secs."""
    query = """
        SELECT DISTINCT r.id AS race_id, r.netkeiba_id, r.date, r.race_name_jp,
            COUNT(res.id) AS total_results,
            SUM(CASE WHEN res.last_3f_secs IS NULL THEN 1 ELSE 0 END) AS missing_last3f
        FROM horsebet.races r
        JOIN horsebet.entries e ON e.race_id = r.id
        JOIN horsebet.results res ON res.entry_id = e.id
        WHERE res.last_3f_secs IS NULL
          AND res.time_secs IS NOT NULL
    """
    params = {}
    if year:
        query += " AND EXTRACT(YEAR FROM r.date) = :year"
        params["year"] = year
    query += """
        GROUP BY r.id, r.netkeiba_id, r.date, r.race_name_jp
        ORDER BY r.date, r.id
    """
    if limit:
        query += " LIMIT :limit"
        params["limit"] = limit

    with get_session() as session:
        rows = session.execute(text(query), params).fetchall()
    return [
        {
            "race_id": r[0],
            "netkeiba_id": r[1],
            "date": str(r[2]),
            "race_name_jp": r[3],
            "total_results": r[4],
            "missing_last3f": r[5],
        }
        for r in rows
    ]


def _process_one_race(race: dict, idx: int, total: int, dry_run: bool) -> dict:
    """Re-scrape a single race and update results with sectional data."""
    stats = {"updated": 0, "skipped": 0, "failed": False}
    nk_id = race["netkeiba_id"]

    # Throttle to avoid rate limiting
    time.sleep(random.uniform(0.5, 1.5))

    url = RESULT_URL.format(race_id=nk_id)
    soup = _fetch(url)
    if not soup:
        log.warning(f"  [{idx}/{total}] ❌ Failed to fetch {race['race_name_jp']} ({nk_id})")
        stats["failed"] = True
        return stats

    race_data = parse_race_page(soup, nk_id)
    if not race_data or not race_data.entries:
        log.warning(f"  [{idx}/{total}] ⏭ No data for {race['race_name_jp']} ({nk_id})")
        stats["failed"] = True
        return stats

    # Count entries that have last_3f data
    entries_with_last3f = sum(1 for e in race_data.entries if e.last_3f_secs is not None)

    if dry_run:
        log.info(
            f"  [{idx}/{total}] 🔍 {race['race_name_jp']} ({race['date']}): "
            f"{entries_with_last3f}/{len(race_data.entries)} have last_3f (dry run)"
        )
        stats["updated"] = entries_with_last3f
        return stats

    with get_session() as session:
        for entry in race_data.entries:
            if entry.last_3f_secs is None:
                stats["skipped"] += 1
                continue

            # Derive first_3f from time_secs - last_3f if we have the data
            first_3f = None
            if entry.time_secs and entry.last_3f_secs:
                first_3f = round(entry.time_secs - entry.last_3f_secs, 1)

            # Update existing result row via entry match
            result = session.execute(
                text("""
                    UPDATE horsebet.results res
                    SET last_3f_secs = COALESCE(:last_3f, res.last_3f_secs),
                        first_3f_secs = COALESCE(:first_3f, res.first_3f_secs),
                        corner_positions = COALESCE(:corners, res.corner_positions)
                    FROM horsebet.entries e
                    WHERE res.entry_id = e.id
                      AND e.race_id = :race_id
                      AND e.post_position = :pp
                      AND res.last_3f_secs IS NULL
                """),
                {
                    "last_3f": entry.last_3f_secs,
                    "first_3f": first_3f,
                    "corners": entry.corner_positions,
                    "race_id": race["race_id"],
                    "pp": entry.post_position,
                },
            )
            if result.rowcount > 0:
                stats["updated"] += result.rowcount

    log.info(
        f"  [{idx}/{total}] ✅ {race['race_name_jp']} ({race['date']}): "
        f"+{stats['updated']} updated, {stats['skipped']} skipped (no last_3f on page)"
    )
    return stats


def backfill_sectionals(
    year: Optional[int] = None,
    dry_run: bool = False,
    workers: int = 3,
    limit: Optional[int] = None,
):
    """Main backfill function."""
    races = get_races_missing_sectionals(year=year, limit=limit)
    total = len(races)
    log.info(f"Found {total} races with missing sectionals (workers={workers}, dry_run={dry_run})")

    if not races:
        log.info("Nothing to backfill!")
        return

    total_updated = 0
    total_skipped = 0
    total_failed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one_race, race, i, total, dry_run): i
            for i, race in enumerate(races, 1)
        }
        for future in as_completed(futures):
            try:
                stats = future.result()
            except Exception as e:
                log.error(f"Worker exception: {e}")
                with lock:
                    total_failed += 1
                continue

            with lock:
                total_updated += stats["updated"]
                total_skipped += stats["skipped"]
                if stats["failed"]:
                    total_failed += 1
                done = total_updated + total_skipped + total_failed
                idx = futures[future]
                if idx % 50 == 0 or idx == total:
                    log.info(
                        f"  Progress: {idx}/{total} "
                        f"(+{total_updated} updated | {total_skipped} skipped | ❌ {total_failed})"
                    )

    log.info(f"\n{'=' * 50}")
    log.info(f"Backfill complete: +{total_updated} updated, {total_skipped} skipped, {total_failed} failed")
    if dry_run:
        log.info("(dry run — no DB changes made)")
    log.info(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Backfill Missing Sectional Times")
    parser.add_argument("--year", type=int, help="Only backfill for this year")
    parser.add_argument("--limit", type=int, help="Limit number of races to process")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't update DB")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent workers (default: 3)")
    args = parser.parse_args()
    backfill_sectionals(year=args.year, dry_run=args.dry_run, workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
