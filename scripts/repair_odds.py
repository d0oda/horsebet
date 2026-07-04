"""
UmaEdge — Repair Odds

Re-scrapes all races from 2024-01-01 onwards to fix the target leakage
where `odds_win` and other result metrics were overwritten by Netkeiba's
new summary tables.

Usage:
    python -m scripts.repair_odds --workers 5
"""

import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import text

from scraper.db import get_session
from scraper.netkeiba import RESULT_URL, _fetch, parse_result_page

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("repair_odds")


def get_races_to_repair() -> list[dict]:
    """Find all races from 2024 onwards to unconditionally repair."""
    query = """
        SELECT DISTINCT r.id AS race_id, r.netkeiba_id, r.date, r.race_name_jp
        FROM races r
        JOIN entries e ON r.id = e.race_id
        JOIN results res ON e.id = res.entry_id
        WHERE r.date >= '2024-01-01' AND e.odds_win >= 100 AND res.finish_pos = 1
        ORDER BY r.date DESC
    """
    with get_session() as session:
        rows = session.execute(text(query)).fetchall()
        
    return [
        {
            "race_id": r[0],
            "netkeiba_id": r[1],
            "date": str(r[2]),
            "race_name_jp": r[3],
        }
        for r in rows
    ]


def _repair_one_race(race: dict, idx: int, total: int) -> dict:
    stats = {"entries_fixed": 0, "failed": False}
    nk_id = race["netkeiba_id"]
    race_id = race["race_id"]

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

    with get_session() as session:
        for entry in race_data.entries:
            if not entry.post_position:
                continue

            # Find the entry_id in DB by matching race_id + post_position
            entry_row = session.execute(
                text("""
                    SELECT e.id FROM entries e
                    WHERE e.race_id = :race_id AND e.post_position = :pp
                """),
                {"race_id": race_id, "pp": entry.post_position},
            ).fetchone()

            if not entry_row:
                continue
            entry_id = entry_row[0]

            # FORCE overwrite odds_win and popularity in entries
            session.execute(
                text("""
                    UPDATE entries
                    SET odds_win = :odds,
                        popularity = :pop
                    WHERE id = :eid
                """),
                {
                    "odds": entry.odds_win,
                    "pop": entry.popularity,
                    "eid": entry_id,
                },
            )

            # FORCE overwrite margin, time_secs, last_3f, corners in results
            session.execute(
                text("""
                    UPDATE results
                    SET margin = :margin,
                        time_secs = :time_secs,
                        last_3f_secs = :last_3f,
                        corner_positions = :corners
                    WHERE entry_id = :eid
                """),
                {
                    "margin": entry.margin,
                    "time_secs": entry.time_secs,
                    "last_3f": entry.last_3f_secs,
                    "corners": entry.corner_positions,
                    "eid": entry_id,
                },
            )
            stats["entries_fixed"] += 1

        session.commit()

    log.info(
        f"  [{idx}/{total}] ✅ {race['date']} {race['race_name_jp']}: "
        f"{stats['entries_fixed']} entries repaired"
    )
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    races = get_races_to_repair()
    total = len(races)
    log.info(f"Found {total} races to repair (using {args.workers} workers)")

    total_fixed = 0
    total_failed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_repair_one_race, race, i, total): i
            for i, race in enumerate(races, 1)
        }
        for future in as_completed(futures):
            stats = future.result()
            with lock:
                total_fixed += stats["entries_fixed"]
                if stats["failed"]:
                    total_failed += 1

    log.info(f"Repair complete: {total_fixed} entries fixed. {total_failed} races failed.")

if __name__ == "__main__":
    main()
