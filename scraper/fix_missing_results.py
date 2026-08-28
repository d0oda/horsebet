#!/usr/bin/env python3
"""
UmaEdge — Fix Missing Results (Legacy DB Archive)

Finds races in the database that have entries but no finish positions
(because they were scraped from the live site after the results were archived).
Fetches the results from db.netkeiba.com and backfills the `results` table.
"""

import logging
import time
from sqlalchemy import text
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from scraper.db import get_session
from scraper.netkeiba import LEGACY_RACE_URL, _parse_legacy_race_page, _fetch, HEADERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Thread-%(threadName)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fix_results")

def fix_race(db_id: int, netkeiba_id: str) -> bool:
    url = LEGACY_RACE_URL.format(race_id=netkeiba_id)
    soup = _fetch(url)
    if not soup:
        return False
        
    race = _parse_legacy_race_page(soup, netkeiba_id)
    if not race or not race.entries:
        return False
        
    has_results = any(e.finish_pos is not None for e in race.entries)
    if not has_results:
        return False
        
    # Map post_position -> entry data to update the DB
    try:
        with get_session() as session:
            # Get existing entry IDs for this race
            entry_rows = session.execute(
                text("SELECT id, post_position FROM entries WHERE race_id = :rid"),
                {"rid": db_id}
            ).fetchall()
            
            entry_map = {r.post_position: r.id for r in entry_rows}
            
            inserted = 0
            for e in race.entries:
                if e.finish_pos is None:
                    continue
                
                entry_id = entry_map.get(e.post_position)
                if not entry_id:
                    continue
                    
                session.execute(
                    text("""
                        INSERT INTO results (
                            entry_id, finish_pos, margin, time_secs,
                            last_3f_secs, corner_positions
                        ) VALUES (
                            :entry_id, :finish_pos, :margin, :time_secs,
                            :last_3f, :corners
                        )
                        ON CONFLICT (entry_id) DO UPDATE SET
                            finish_pos = EXCLUDED.finish_pos,
                            margin = EXCLUDED.margin,
                            time_secs = EXCLUDED.time_secs,
                            last_3f_secs = EXCLUDED.last_3f_secs,
                            corner_positions = EXCLUDED.corner_positions
                    """),
                    {
                        "entry_id": entry_id,
                        "finish_pos": e.finish_pos,
                        "margin": e.margin,
                        "time_secs": e.time_secs,
                        "last_3f": e.last_3f_secs,
                        "corners": e.corner_positions,
                    }
                )
                session.execute(
                    text("""
                        UPDATE entries
                        SET finish_pos = COALESCE(:finish_pos, finish_pos),
                            margin = COALESCE(:margin, margin),
                            time_secs = COALESCE(:time_secs, time_secs),
                            last_3f_secs = COALESCE(:last_3f, last_3f_secs),
                            corner_positions = COALESCE(:corners, corner_positions)
                        WHERE id = :entry_id
                    """),
                    {
                        "entry_id": entry_id,
                        "finish_pos": e.finish_pos,
                        "margin": e.margin,
                        "time_secs": e.time_secs,
                        "last_3f": e.last_3f_secs,
                        "corners": e.corner_positions,
                    }
                )
                inserted += 1
                
            session.commit()
            if inserted > 0:
                log.info(f"Fixed race {netkeiba_id} — Backfilled {inserted} results")
                return True
    except Exception as e:
        log.error(f"Failed updating DB for {netkeiba_id}: {e}")
        
    return False

def main():
    log.info("Finding races with missing results...")
    
    with get_session() as session:
        # Find races that have entries, but no finish positions in the results table
        query = text("""
            SELECT r.id, r.netkeiba_id 
            FROM races r
            WHERE EXISTS (SELECT 1 FROM entries e WHERE e.race_id = r.id)
            AND NOT EXISTS (
                SELECT 1 FROM entries e 
                JOIN results res ON e.id = res.entry_id 
                WHERE e.race_id = r.id AND res.finish_pos IS NOT NULL
            )
            ORDER BY r.date DESC
        """)
        races_to_fix = session.execute(query).fetchall()
        
    log.info(f"Found {len(races_to_fix)} races missing results. Starting backfill...")
    
    if not races_to_fix:
        return
        
    success = 0
    # Use 4 workers to backfill quickly
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="FixWorker") as executor:
        futures = [executor.submit(fix_race, r.id, r.netkeiba_id) for r in races_to_fix]
        for future in as_completed(futures):
            if future.result():
                success += 1
                
    log.info(f"Done! Successfully backfilled results for {success} out of {len(races_to_fix)} races.")

if __name__ == "__main__":
    main()
