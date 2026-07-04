import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text
from scraper.db import get_session
from scraper.netkeiba import RESULT_URL, _fetch, parse_result_page
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("fix_winners")

def get_races_to_repair() -> list[dict]:
    query = """
        SELECT DISTINCT r.id AS race_id, r.netkeiba_id, r.date, r.race_name_jp
        FROM races r
        JOIN entries e ON r.id = e.race_id
        JOIN results res ON e.id = res.entry_id
        WHERE r.date >= '2024-01-01' AND res.finish_pos = 1
          AND e.odds_win >= 10.0 AND e.odds_win <= 13.0
        ORDER BY r.date DESC
    """
    with get_session() as session:
        rows = session.execute(text(query)).fetchall()
        
    return [{"race_id": r[0], "netkeiba_id": r[1], "date": str(r[2]), "race_name_jp": r[3]} for r in rows]

def _repair_one_race(race: dict, idx: int, total: int) -> dict:
    stats = {"fixed": 0, "failed": False}
    nk_id = race["netkeiba_id"]
    race_id = race["race_id"]

    url = RESULT_URL.format(race_id=nk_id)
    soup = _fetch(url)
    if not soup:
        log.warning(f"  [{idx}/{total}] ❌ Failed fetch {nk_id}")
        stats["failed"] = True
        return stats

    race_data = parse_result_page(soup, nk_id)
    if not race_data or not race_data.entries:
        stats["failed"] = True
        return stats

    with get_session() as session:
        for entry in race_data.entries:
            if not entry.post_position or not entry.odds_win:
                continue

            # Only update the winner
            if entry.finish_pos != 1:
                continue

            entry_row = session.execute(
                text("SELECT e.id, e.odds_win FROM entries e WHERE e.race_id = :race_id AND e.post_position = :pp"),
                {"race_id": race_id, "pp": entry.post_position},
            ).fetchone()

            if not entry_row:
                continue
            entry_id, old_odds = entry_row[0], entry_row[1]

            if abs(old_odds - entry.odds_win) > 0.1:
                session.execute(
                    text("UPDATE entries SET odds_win = :odds, popularity = :pop WHERE id = :eid"),
                    {"odds": entry.odds_win, "pop": entry.popularity, "eid": entry_id},
                )
                stats["fixed"] += 1

        session.commit()

    if stats["fixed"] > 0:
        log.info(f"  [{idx}/{total}] ✅ Fixed {race['date']} {race['race_name_jp']}")
    return stats

def main():
    races = get_races_to_repair()
    total = len(races)
    log.info(f"Found {total} races that might have corrupted winner odds (odds between 10.0 and 13.0)")
    
    total_fixed, total_failed = 0, 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_repair_one_race, race, i, total): i for i, race in enumerate(races, 1)}
        for future in as_completed(futures):
            stats = future.result()
            with lock:
                total_fixed += stats["fixed"]
                if stats["failed"]: total_failed += 1

    log.info(f"Repair complete: {total_fixed} winners fixed. {total_failed} fetches failed.")

if __name__ == "__main__":
    main()
