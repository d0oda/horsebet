"""
UmaEdge — Backfill Missing Race Metadata (going, weather) and Result Data (corners, running_style).

Re-scrapes netkeiba race pages for races missing going/weather and results missing
corner_positions/running_style. Updates existing rows.

Usage:
    python -m scraper.backfill_race_metadata                    # backfill all
    python -m scraper.backfill_race_metadata --dry-run          # preview only
    python -m scraper.backfill_race_metadata --workers 3        # concurrency
"""

import argparse
import logging
import random
import re
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
from scraper.backfill_running_style import infer_running_style

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_race_metadata")


def get_races_needing_backfill() -> list[dict]:
    """Find races missing going/weather, or with results missing running_style."""
    query = """
        SELECT r.id AS race_id, r.netkeiba_id, r.date, r.race_name_jp,
            r.going IS NULL AS missing_going,
            r.weather IS NULL AS missing_weather,
            r.field_size,
            (SELECT COUNT(*) FROM entries e2
             JOIN results res2 ON res2.entry_id = e2.id
             WHERE e2.race_id = r.id AND res2.running_style IS NULL) AS missing_styles
        FROM races r
        WHERE r.going IS NULL
           OR r.weather IS NULL
           OR EXISTS (
               SELECT 1 FROM entries e
               JOIN results res ON res.entry_id = e.id
               WHERE e.race_id = r.id AND res.running_style IS NULL
           )
        ORDER BY r.date, r.id
    """
    with get_session() as session:
        rows = session.execute(text(query)).fetchall()

    return [
        {
            "race_id": r[0],
            "netkeiba_id": r[1],
            "date": str(r[2]),
            "race_name_jp": r[3],
            "missing_going": r[4],
            "missing_weather": r[5],
            "field_size": r[6],
            "missing_styles": r[7],
        }
        for r in rows
    ]


def _process_one_race(race: dict, idx: int, total: int, dry_run: bool) -> dict:
    """Re-scrape a single race and update race metadata + result data."""
    stats = {"going_updated": 0, "weather_updated": 0, "corners_updated": 0, "styles_updated": 0, "failed": False}
    nk_id = race["netkeiba_id"]

    time.sleep(random.uniform(0.5, 1.5))

    url = RESULT_URL.format(race_id=nk_id)
    soup = _fetch(url)
    if not soup:
        log.warning(f"  [{idx}/{total}] ❌ Failed to fetch {nk_id}")
        stats["failed"] = True
        return stats

    race_data = parse_race_page(soup, nk_id)
    if not race_data:
        log.warning(f"  [{idx}/{total}] ⏭ No data for {nk_id}")
        stats["failed"] = True
        return stats

    # Fallback: if parser didn't capture going, try extracting from raw page
    if not race_data.going:
        import re
        page_text = soup.get_text()
        going_match = re.search(r"馬場[：:]\s*(良|稍重|稍|不良|不|重)", page_text)
        if going_match:
            val = going_match.group(1)
            race_data.going = {"稍": "稍重", "不": "不良"}.get(val, val)

    if dry_run:
        parts = []
        if race["missing_going"] and race_data.going:
            parts.append(f"going={race_data.going}")
        if race["missing_weather"] and race_data.weather:
            parts.append(f"weather={race_data.weather}")
        entries_with_corners = sum(1 for e in race_data.entries if e.corner_positions)
        if entries_with_corners:
            parts.append(f"corners={entries_with_corners}")
        log.info(f"  [{idx}/{total}] 🔍 {race['race_name_jp']} ({race['date']}): {', '.join(parts) or 'no new data'}")
        stats["going_updated"] = 1 if race["missing_going"] and race_data.going else 0
        stats["weather_updated"] = 1 if race["missing_weather"] and race_data.weather else 0
        stats["corners_updated"] = entries_with_corners
        return stats

    with get_session() as session:
        # Update race-level metadata (going, weather)
        updates = {}
        if race["missing_going"] and race_data.going:
            updates["going"] = race_data.going
            stats["going_updated"] = 1
        if race["missing_weather"] and race_data.weather:
            updates["weather"] = race_data.weather
            stats["weather_updated"] = 1

        if updates:
            set_clauses = ", ".join(f"{k} = :{k}" for k in updates)
            updates["race_id"] = race["race_id"]
            session.execute(
                text(f"UPDATE races SET {set_clauses} WHERE id = :race_id"),
                updates,
            )

        # Update result-level data (corner_positions, running_style)
        for entry in race_data.entries:
            if not entry.corner_positions:
                continue

            # Derive running_style from corner positions
            style = infer_running_style(entry.corner_positions, race["field_size"] or len(race_data.entries))

            result = session.execute(
                text("""
                    UPDATE results res
                    SET corner_positions = COALESCE(:corners, res.corner_positions),
                        running_style = COALESCE(:style, res.running_style)
                    FROM entries e
                    WHERE res.entry_id = e.id
                      AND e.race_id = :race_id
                      AND e.post_position = :pp
                      AND (res.corner_positions IS NULL OR res.running_style IS NULL)
                """),
                {
                    "corners": entry.corner_positions,
                    "style": style,
                    "race_id": race["race_id"],
                    "pp": entry.post_position,
                },
            )
            if result.rowcount > 0:
                stats["corners_updated"] += result.rowcount
                if style:
                    stats["styles_updated"] += result.rowcount

    parts = []
    if stats["going_updated"]:
        parts.append(f"going={race_data.going}")
    if stats["weather_updated"]:
        parts.append(f"weather={race_data.weather}")
    if stats["corners_updated"]:
        parts.append(f"+{stats['corners_updated']} corners/styles")
    log.info(f"  [{idx}/{total}] ✅ {race['race_name_jp']} ({race['date']}): {', '.join(parts) or 'no updates'}")
    return stats


def backfill_race_metadata(dry_run: bool = False, workers: int = 3):
    """Main backfill function."""
    races = get_races_needing_backfill()
    total = len(races)
    log.info(f"Found {total} races needing metadata backfill (workers={workers})")

    if not races:
        log.info("Nothing to backfill!")
        return

    totals = {"going": 0, "weather": 0, "corners": 0, "styles": 0, "failed": 0}
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
                    totals["failed"] += 1
                continue

            with lock:
                totals["going"] += stats["going_updated"]
                totals["weather"] += stats["weather_updated"]
                totals["corners"] += stats["corners_updated"]
                totals["styles"] += stats["styles_updated"]
                if stats["failed"]:
                    totals["failed"] += 1
                idx = futures[future]
                if idx % 50 == 0 or idx == total:
                    log.info(
                        f"  Progress: {idx}/{total} "
                        f"(going: +{totals['going']} | weather: +{totals['weather']} | "
                        f"corners: +{totals['corners']} | styles: +{totals['styles']} | ❌ {totals['failed']})"
                    )

    log.info(f"\n{'=' * 50}")
    log.info(f"Backfill complete:")
    log.info(f"  going: +{totals['going']}")
    log.info(f"  weather: +{totals['weather']}")
    log.info(f"  corners: +{totals['corners']}")
    log.info(f"  styles: +{totals['styles']}")
    log.info(f"  failed: {totals['failed']}")
    if dry_run:
        log.info("(dry run — no DB changes made)")
    log.info(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Backfill Race Metadata")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent workers (default: 3)")
    args = parser.parse_args()
    backfill_race_metadata(dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()
