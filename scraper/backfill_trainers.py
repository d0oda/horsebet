"""
UmaEdge — Backfill Trainers & Class for Existing Races.

Re-scrapes already-stored races to extract trainer names and race class,
then updates the database without re-inserting entries/results.
Resumable: only processes races that still have horses with NULL trainer_id.

Usage:
    python -m scraper.backfill_trainers --max 50
    python -m scraper.backfill_trainers            # all remaining
    python -m scraper.backfill_trainers --workers 3
"""

import argparse
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

from scraper.db import get_session
from scraper.netkeiba import (
    RACE_URL,
    HEADERS,
    JRA_COURSES,
    _extract_race_class,
    parse_race_page,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_trainers")


def _fetch_fast(url: str, retries: int = 3):
    """Fetch with reduced delay for backfill speed."""
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(0.5, 1.2))
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.encoding = "EUC-JP"
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            elif resp.status_code == 404:
                return None
            else:
                log.warning(f"HTTP {resp.status_code} for {url} (attempt {attempt + 1})")
        except requests.RequestException as e:
            log.warning(f"Request error: {e} (attempt {attempt + 1})")
        if attempt < retries - 1:
            time.sleep(3 * (attempt + 1))
    return None


def backfill_race(race_nk_id: str, race_db_id: int, session) -> dict:
    """
    Re-scrape a single race and backfill trainer_id for its horses
    and class for the race.

    Returns dict with counts of updates made.
    """
    stats = {"trainers_linked": 0, "class_updated": 0}

    url = RACE_URL.format(race_id=race_nk_id)
    soup = _fetch_fast(url)
    if not soup:
        return stats

    race = parse_race_page(soup, race_nk_id)
    if not race or not race.entries:
        return stats

    # Update race class if missing
    if race.class_:
        result = session.execute(
            text("UPDATE races SET class = :cls WHERE id = :rid AND class IS NULL"),
            {"cls": race.class_, "rid": race_db_id},
        )
        if result.rowcount > 0:
            stats["class_updated"] = 1

    # Update trainer_id for each horse
    for entry in race.entries:
        if not entry.trainer_name_jp:
            continue

        horse_nk_id = entry.horse.netkeiba_id
        if not horse_nk_id or horse_nk_id.startswith("unknown_"):
            continue

        # Upsert trainer
        trainer_result = session.execute(
            text("""
                INSERT INTO trainers (name, name_jp)
                VALUES (:name, :name_jp)
                ON CONFLICT (name_jp) DO NOTHING
                RETURNING id
            """),
            {"name": entry.trainer_name_jp, "name_jp": entry.trainer_name_jp},
        ).fetchone()

        if trainer_result:
            trainer_id = trainer_result[0]
        else:
            trainer_id = session.execute(
                text("SELECT id FROM trainers WHERE name_jp = :name_jp"),
                {"name_jp": entry.trainer_name_jp},
            ).scalar()

        if trainer_id:
            result = session.execute(
                text("""
                    UPDATE horses SET trainer_id = :tid
                    WHERE netkeiba_id = :nkid AND trainer_id IS NULL
                """),
                {"tid": trainer_id, "nkid": horse_nk_id},
            )
            if result.rowcount > 0:
                stats["trainers_linked"] += 1

    return stats


def _process_race(race_db_id, race_nk_id, idx, total):
    """Process a single race. Returns stats dict."""
    with get_session() as session:
        stats = backfill_race(race_nk_id, race_db_id, session)
    if stats["trainers_linked"] > 0 or stats["class_updated"] > 0:
        log.info(
            f"  [{idx}/{total}] race {race_nk_id}: "
            f"+{stats['trainers_linked']} trainers, +{stats['class_updated']} class"
        )
    return stats


def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Backfill Trainers & Class")
    parser.add_argument("--max", type=int, help="Max races to process")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent workers (default: 3)")
    args = parser.parse_args()

    with get_session() as session:
        # Only get races that still have horses with NULL trainer_id (resumable)
        query = """
            SELECT DISTINCT r.id, r.netkeiba_id
            FROM races r
            JOIN entries e ON e.race_id = r.id
            JOIN horses h ON h.id = e.horse_id
            WHERE h.trainer_id IS NULL
            ORDER BY r.id
        """
        races = session.execute(text(query)).fetchall()

    total = len(races)
    if args.max:
        races = races[:args.max]

    log.info(f"Backfilling {len(races)} races with missing trainers (using {args.workers} workers)...")

    total_trainers = 0
    total_class = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_race, db_id, nk_id, i + 1, len(races)): i
            for i, (db_id, nk_id) in enumerate(races)
        }
        for future in as_completed(futures):
            stats = future.result()
            with lock:
                total_trainers += stats["trainers_linked"]
                total_class += stats["class_updated"]
                done = futures[future] + 1
                if done % 25 == 0:
                    log.info(
                        f"  Progress: {done}/{len(races)} | "
                        f"trainers: +{total_trainers}, class: +{total_class}"
                    )

    log.info(f"\n✅ Backfill complete: {total_trainers} trainer links, {total_class} class updates")


if __name__ == "__main__":
    main()
