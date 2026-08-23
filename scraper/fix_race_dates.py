"""
UmaEdge — Fix Race Dates.

Fixes incorrect race dates caused by batch_scrape.py deriving dates from
race IDs (treating venue+meeting as month+day). Re-fetches the actual date
from db.netkeiba.com/race/{race_id}/ for each affected race.

Usage:
    python -m scraper.fix_race_dates                # fix all bad dates
    python -m scraper.fix_race_dates --dry-run      # preview only
    python -m scraper.fix_race_dates --workers 5    # concurrency
"""

import argparse
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

from scraper.db import get_session
from scraper.netkeiba import HEADERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fix_race_dates")

DB_RACE_URL = "https://db.netkeiba.com/race/{race_id}/"


def get_races_with_bad_dates() -> list[dict]:
    """Find races where date appears to be derived from venue/meeting IDs."""
    with get_session() as session:
        # Bad dates: dates that have way too many races (>36 = more than 3 venues)
        bad_dates = session.execute(text("""
            SELECT date FROM races
            GROUP BY date
            HAVING COUNT(*) > 36
        """)).fetchall()
        bad_date_set = {str(r[0]) for r in bad_dates}

        if not bad_date_set:
            return []

        races = session.execute(text("""
            SELECT id, netkeiba_id, CAST(date AS TEXT), race_name_jp
            FROM races
            WHERE date IN (SELECT date FROM races
                           GROUP BY date HAVING COUNT(*) > 36)
            ORDER BY netkeiba_id
        """)).fetchall()

    return [
        {"id": r[0], "netkeiba_id": r[1], "old_date": r[2], "name": r[3]}
        for r in races
    ]


def fetch_actual_date(race_id: str) -> str | None:
    """Fetch the actual race date from db.netkeiba.com race page."""
    url = DB_RACE_URL.format(race_id=race_id)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.encoding = "EUC-JP"
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")

        # Look for date pattern: YYYY年M月D日
        text_content = soup.get_text()
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text_content)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    except Exception as e:
        log.warning(f"  Error fetching {race_id}: {e}")
    return None


def process_one(race: dict, idx: int, total: int, dry_run: bool) -> dict:
    """Fix date for one race."""
    nk_id = race["netkeiba_id"]
    actual_date = fetch_actual_date(nk_id)

    if not actual_date:
        return {"fixed": False, "failed": True}

    if actual_date == race["old_date"]:
        return {"fixed": False, "failed": False}  # already correct

    if dry_run:
        if idx <= 20 or idx % 100 == 0:
            log.info(f"  [{idx}/{total}] {race['name']}: "
                     f"{race['old_date']} → {actual_date}")
        return {"fixed": True, "failed": False}

    with get_session() as session:
        session.execute(
            text("UPDATE races SET date = :new_date WHERE id = :rid"),
            {"new_date": actual_date, "rid": race["id"]},
        )

    return {"fixed": True, "failed": False}


def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Fix Race Dates")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    races = get_races_with_bad_dates()
    total = len(races)
    log.info(f"Found {total} races with potentially bad dates")

    if not races:
        log.info("Nothing to fix!")
        return

    fixed = 0
    failed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, race, i, total, args.dry_run): i
            for i, race in enumerate(races, 1)
        }
        for future in as_completed(futures):
            result = future.result()
            with lock:
                if result["fixed"]:
                    fixed += 1
                if result["failed"]:
                    failed += 1
                done = fixed + failed + (futures[future] - fixed - failed)
                if done % 200 == 0:
                    log.info(f"  Progress: {done}/{total} "
                             f"(fixed: {fixed} | failed: {failed})")

    log.info(f"\n{'=' * 50}")
    log.info(f"Date fix complete: {fixed} fixed, {failed} failed, "
             f"{total - fixed - failed} already correct")
    if args.dry_run:
        log.info("(dry run — no changes made)")
    log.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
