"""
Backfill sire_name for existing horses by scraping their netkeiba profile pages.

Usage:
    python -m scraper.backfill_sires                # backfill all missing
    python -m scraper.backfill_sires --limit 100    # backfill first 100
    python -m scraper.backfill_sires --dry-run      # parse only, don't update DB
    python -m scraper.backfill_sires --workers 3    # concurrency (default 3)
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_sires")

HORSE_PED_URL = "https://db.netkeiba.com/horse/ped/{horse_nk_id}/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}


def _fetch(url: str, retries: int = 4) -> Optional[BeautifulSoup]:
    """Fetch a URL with retries, polite delay, and rate-limit backoff."""
    for attempt in range(retries):
        try:
            delay = random.uniform(1.0, 2.0)
            time.sleep(delay)
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.encoding = "EUC-JP"
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            elif resp.status_code == 404:
                log.debug(f"404: {url}")
                return None
            elif resp.status_code in (400, 403, 429, 503):
                # Rate limited — exponential backoff
                backoff = min(60, 15 * (2 ** attempt))
                log.warning(f"Rate limited ({resp.status_code}), backing off {backoff}s (attempt {attempt + 1})")
                time.sleep(backoff)
            else:
                log.warning(f"HTTP {resp.status_code} for {url} (attempt {attempt + 1})")
        except requests.RequestException as e:
            log.warning(f"Request error: {e} (attempt {attempt + 1})")
        if attempt < retries - 1:
            time.sleep(5 * (attempt + 1))
    return None


def parse_sire_name(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract sire name from a netkeiba horse PEDIGREE page (/horse/ped/{id}/).
    
    The pedigree table has class 'blood_table' with the sire in
    row 0, column 0.
    """
    blood_table = soup.find("table", class_="blood_table")
    if blood_table:
        rows = blood_table.find_all("tr")
        if rows:
            first_cell = rows[0].find("td")
            if first_cell:
                link = first_cell.find("a")
                if link:
                    return link.get_text(strip=True)
                text_content = first_cell.get_text(strip=True)
                if text_content:
                    return text_content
    return None


def get_horses_missing_sire(limit: Optional[int] = None) -> list[dict]:
    """Get horses that don't have sire_name populated."""
    query = """
        SELECT id, netkeiba_id, name_jp
        FROM horses
        WHERE sire_name IS NULL
          AND netkeiba_id IS NOT NULL
          AND netkeiba_id NOT LIKE 'unknown_%'
        ORDER BY id
    """
    if limit:
        query += f" LIMIT {limit}"

    with get_session() as session:
        rows = session.execute(text(query)).fetchall()
    return [{"id": r[0], "netkeiba_id": r[1], "name_jp": r[2]} for r in rows]


def _process_one_horse(horse: dict, idx: int, total: int, dry_run: bool) -> str:
    """Process a single horse. Returns 'updated', 'failed', or 'skipped'."""
    nk_id = horse["netkeiba_id"]
    url = HORSE_PED_URL.format(horse_nk_id=nk_id)

    soup = _fetch(url)
    if not soup:
        log.warning(f"  [{idx}/{total}] ❌ Failed to fetch {horse['name_jp']} ({nk_id})")
        return "failed"

    sire_name = parse_sire_name(soup)
    if not sire_name:
        log.debug(f"  [{idx}/{total}] ⏭ No sire found for {horse['name_jp']} ({nk_id})")
        return "skipped"

    if dry_run:
        log.info(f"  [{idx}/{total}] 🔍 {horse['name_jp']} → sire: {sire_name} (dry run)")
        return "updated"

    # Update DB
    with get_session() as session:
        session.execute(
            text("UPDATE horses SET sire_name = :sire WHERE id = :hid"),
            {"sire": sire_name, "hid": horse["id"]},
        )
    log.info(f"  [{idx}/{total}] ✅ {horse['name_jp']} → sire: {sire_name}")
    return "updated"


def backfill_sires(limit: Optional[int] = None, dry_run: bool = False,
                   workers: int = 3):
    """Main backfill function with thread-pool concurrency."""
    horses = get_horses_missing_sire(limit)
    total = len(horses)
    log.info(f"Found {total} horses missing sire_name (using {workers} workers)")

    if not horses:
        log.info("Nothing to backfill!")
        return

    updated = 0
    failed = 0
    skipped = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one_horse, horse, i, total, dry_run): i
            for i, horse in enumerate(horses, 1)
        }
        for future in as_completed(futures):
            result = future.result()
            with lock:
                if result == "updated":
                    updated += 1
                elif result == "failed":
                    failed += 1
                else:
                    skipped += 1
                done = updated + failed + skipped
                if done % 25 == 0:
                    log.info(
                        f"  Progress: {done}/{total} "
                        f"(✅ {updated} | ❌ {failed} | ⏭ {skipped})"
                    )

    log.info(f"\n{'=' * 50}")
    log.info(f"Backfill complete: {updated} updated, {failed} failed, {skipped} no sire found")
    log.info(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Backfill sire names from netkeiba horse profiles")
    parser.add_argument("--limit", type=int, default=None, help="Max horses to process")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, don't update DB")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent workers (default: 3)")
    args = parser.parse_args()
    backfill_sires(limit=args.limit, dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()
