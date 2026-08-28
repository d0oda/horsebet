"""
UmaEdge — Backfill Results for Finished Races.

Re-scrapes finished races that have entry rows but no corresponding result rows.
Inserts into the `results` table and also updates entries with final result data.

Usage:
    python -m scraper.backfill_results                   # backfill all missing
    python -m scraper.backfill_results --date 2026-03-01 # specific date only
    python -m scraper.backfill_results --dry-run         # preview only
    python -m scraper.backfill_results --workers 3       # concurrency
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
    HEADERS,
    _fetch,
    parse_result_page,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_results")


def get_races_missing_results(date: Optional[str] = None) -> list[dict]:
    """Find races that have entries but are missing result rows."""
    query = """
        SELECT DISTINCT r.id AS race_id, r.netkeiba_id, r.date, r.race_name_jp,
            COUNT(e.id) AS total_entries,
            COUNT(res.id) AS results_found
        FROM races r
        JOIN entries e ON e.race_id = r.id
        LEFT JOIN results res ON res.entry_id = e.id
    """
    params = {}
    if date:
        query += " WHERE r.date = :date"
        params["date"] = date
    query += """
        GROUP BY r.id, r.netkeiba_id, r.date, r.race_name_jp
        HAVING COUNT(res.id) < COUNT(e.id) OR COUNT(e.finish_pos) < COUNT(e.id)
        ORDER BY r.date, r.id
    """

    with get_session() as session:
        rows = session.execute(text(query), params).fetchall()
    return [
        {
            "race_id": r[0],
            "netkeiba_id": r[1],
            "date": str(r[2]),
            "race_name_jp": r[3],
            "total_entries": r[4],
            "results_found": r[5],
        }
        for r in rows
    ]


def _process_one_race(race: dict, idx: int, total: int, dry_run: bool) -> dict:
    """Re-scrape a single race and insert results. Returns stats dict."""
    stats = {"results_inserted": 0, "entries_updated": 0, "failed": False}
    nk_id = race["netkeiba_id"]

    url = RESULT_URL.format(race_id=nk_id)
    soup = _fetch(url)
    if not soup:
        log.warning(f"  [{idx}/{total}] ❌ Failed to fetch {race['race_name_jp']} ({nk_id})")
        stats["failed"] = True
        return stats

    race_data = parse_result_page(soup, nk_id)
    if not race_data or not race_data.entries:
        log.warning(f"  [{idx}/{total}] ⏭ No result data for {race['race_name_jp']} ({nk_id})")
        stats["failed"] = True
        return stats

    if dry_run:
        entries_with_results = sum(1 for e in race_data.entries if e.finish_pos is not None)
        log.info(
            f"  [{idx}/{total}] 🔍 {race['race_name_jp']}: "
            f"{entries_with_results}/{len(race_data.entries)} entries have results (dry run)"
        )
        stats["results_inserted"] = entries_with_results
        return stats

    with get_session() as session:
        for entry in race_data.entries:
            if entry.finish_pos is None and entry.time_secs is None:
                continue

            horse_nk_id = entry.horse.netkeiba_id
            if not horse_nk_id or horse_nk_id.startswith("unknown_"):
                continue

            # Find the entry_id in DB by matching race_id + post_position
            entry_row = session.execute(
                text("""
                    SELECT e.id FROM entries e
                    WHERE e.race_id = :race_id AND e.post_position = :pp
                """),
                {"race_id": race["race_id"], "pp": entry.post_position},
            ).fetchone()

            if not entry_row:
                continue
            entry_id = entry_row[0]

            # Insert into results table (ON CONFLICT skip)
            result = session.execute(
                text("""
                    INSERT INTO results (
                        entry_id, finish_pos, margin, time_secs,
                        last_3f_secs, corner_positions
                    ) VALUES (
                        :entry_id, :finish_pos, :margin, :time_secs,
                        :last_3f, :corners
                    )
                    ON CONFLICT (entry_id) DO UPDATE SET
                        finish_pos = COALESCE(EXCLUDED.finish_pos, results.finish_pos),
                        margin = COALESCE(EXCLUDED.margin, results.margin),
                        time_secs = COALESCE(EXCLUDED.time_secs, results.time_secs),
                        last_3f_secs = COALESCE(EXCLUDED.last_3f_secs, results.last_3f_secs),
                        corner_positions = COALESCE(EXCLUDED.corner_positions, results.corner_positions)
                """),
                {
                    "entry_id": entry_id,
                    "finish_pos": entry.finish_pos,
                    "margin": entry.margin,
                    "time_secs": entry.time_secs,
                    "last_3f": entry.last_3f_secs,
                    "corners": entry.corner_positions,
                },
            )
            if result.rowcount > 0:
                stats["results_inserted"] += 1

            # Also update entries table with result data
            session.execute(
                text("""
                    UPDATE entries
                    SET finish_pos = COALESCE(:fp, finish_pos),
                        time_secs = COALESCE(:ts, time_secs),
                        last_3f_secs = COALESCE(:l3f, last_3f_secs),
                        corner_positions = COALESCE(:cp, corner_positions),
                        margin = COALESCE(:margin, margin),
                        odds_win = COALESCE(:odds, odds_win),
                        popularity = COALESCE(:pop, popularity)
                    WHERE id = :eid
                """),
                {
                    "fp": entry.finish_pos,
                    "ts": entry.time_secs,
                    "l3f": entry.last_3f_secs,
                    "cp": entry.corner_positions,
                    "margin": entry.margin,
                    "odds": entry.odds_win,
                    "pop": entry.popularity,
                    "eid": entry_id,
                },
            )
            stats["entries_updated"] += 1

    log.info(
        f"  [{idx}/{total}] ✅ {race['race_name_jp']}: "
        f"+{stats['results_inserted']} results, {stats['entries_updated']} entries updated"
    )
    return stats


def backfill_results(date: Optional[str] = None, dry_run: bool = False,
                     workers: int = 3):
    """Main backfill function."""
    races = get_races_missing_results(date)
    total = len(races)
    log.info(f"Found {total} races with missing results (using {workers} workers)")

    if not races:
        log.info("Nothing to backfill!")
        return

    total_results = 0
    total_entries = 0
    total_failed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one_race, race, i, total, dry_run): i
            for i, race in enumerate(races, 1)
        }
        for future in as_completed(futures):
            stats = future.result()
            with lock:
                total_results += stats["results_inserted"]
                total_entries += stats["entries_updated"]
                if stats["failed"]:
                    total_failed += 1
                done = futures[future]
                if done % 10 == 0:
                    log.info(
                        f"  Progress: {done}/{total} "
                        f"(results: +{total_results} | entries: +{total_entries} | ❌ {total_failed})"
                    )

    log.info(f"\n{'=' * 50}")
    log.info(f"Backfill complete: +{total_results} results, +{total_entries} entries, {total_failed} failed")
    if dry_run:
        log.info("(dry run — no DB changes made)")
    log.info(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Backfill Missing Results")
    parser.add_argument("--date", type=str, help="Only backfill results for this date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't update DB")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent workers (default: 3)")
    args = parser.parse_args()
    backfill_results(date=args.date, dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()
