"""
UmaEdge — Backfill Sire Names from Umanity.jp

Umanity.jp has individual horse profile pages with pedigree data, using the
same horse codes as netkeiba (our `netkeiba_id` field). This is much faster
than the JRA race-by-race approach since it's 1 HTTP GET per horse.

URL format: https://umanity.jp/en/racedata/db/horse_top.php?code={netkeiba_id}
Sire appears as the first link after ">> Detail" on the page.

Usage:
    python -m scraper.backfill_sires_umanity                # backfill all missing
    python -m scraper.backfill_sires_umanity --limit 100    # first 100
    python -m scraper.backfill_sires_umanity --dry-run      # preview only
    python -m scraper.backfill_sires_umanity --workers 4    # concurrency
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
log = logging.getLogger("backfill_sires_umanity")

HORSE_URL = "https://umanity.jp/en/racedata/db/horse_top.php?code={code}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch a URL with retries and polite delay."""
    for attempt in range(retries):
        try:
            delay = random.uniform(0.3, 0.8)
            time.sleep(delay)
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            elif resp.status_code == 404:
                return None
            elif resp.status_code in (429, 503):
                backoff = min(60, 10 * (2 ** attempt))
                log.warning(f"Rate limited ({resp.status_code}), backing off {backoff}s")
                time.sleep(backoff)
            else:
                log.warning(f"HTTP {resp.status_code} for {url} (attempt {attempt + 1})")
        except requests.RequestException as e:
            log.warning(f"Request error: {e} (attempt {attempt + 1})")
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    return None


def parse_sire_name(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract sire name from a umanity horse profile page.
    
    The page structure has a ">> Detail" link to the pedigree page,
    followed by the sire link. The sire link URL contains
    'horse_top.php?code=' and appears right after the pedigree detail link.
    
    We look for the pedigree section which has links in this order:
    1. ">> Detail" (pedigree page link)
    2. Sire name link (horse_top.php)
    3. "Pedigree" link for sire
    4. "Progeny" link for sire
    5. Sire's sire link
    6. Sire's dam link
    7. Dam name link
    ...
    """
    # Find the ">> Detail" link that points to pedigree page
    detail_link = soup.find("a", string=re.compile(r"Detail"))
    if not detail_link:
        # Try alternate: find link to horse_pedigree.php
        detail_link = soup.find("a", href=re.compile(r"horse_pedigree\.php"))
    
    if not detail_link:
        return None
    
    # The sire link is the very next <a> tag after the Detail link
    # that points to horse_top.php
    next_elem = detail_link
    for _ in range(5):  # Look through next few siblings
        next_elem = next_elem.find_next("a")
        if not next_elem:
            break
        href = next_elem.get("href", "")
        if "horse_top.php" in href and "code=" in href:
            sire_name = next_elem.get_text(strip=True)
            if sire_name and sire_name not in ("Pedigree", "Progeny", "Son/Daughter"):
                return sire_name
    
    return None


def get_horses_missing_sire(limit: Optional[int] = None) -> list[dict]:
    """Get horses missing sire_name with valid netkeiba IDs."""
    query = """
        SELECT id, netkeiba_id, name_jp
        FROM horsebet.horses
        WHERE sire_name IS NULL
          AND netkeiba_id IS NOT NULL
          AND netkeiba_id NOT LIKE 'unknown_%'
          AND netkeiba_id NOT LIKE 'jra_%'
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
    url = HORSE_URL.format(code=nk_id)

    soup = _fetch(url)
    if not soup:
        log.debug(f"  [{idx}/{total}] ❌ Failed to fetch {horse['name_jp']} ({nk_id})")
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
            text("UPDATE horsebet.horses SET sire_name = :sire WHERE id = :hid"),
            {"sire": sire_name, "hid": horse["id"]},
        )
    log.debug(f"  [{idx}/{total}] ✅ {horse['name_jp']} → sire: {sire_name}")
    return "updated"


def backfill_sires(limit: Optional[int] = None, dry_run: bool = False,
                   workers: int = 4):
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
                if done % 50 == 0:
                    log.info(
                        f"  Progress: {done}/{total} "
                        f"(✅ {updated} | ❌ {failed} | ⏭ {skipped})"
                    )

    log.info(f"\n{'=' * 50}")
    log.info(f"Backfill complete: {updated} updated, {failed} failed, {skipped} no sire found")
    log.info(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(
        description="UmaEdge — Backfill sire names from umanity.jp"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max horses to process")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, don't update DB")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent workers (default: 4)")
    args = parser.parse_args()
    backfill_sires(limit=args.limit, dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()
