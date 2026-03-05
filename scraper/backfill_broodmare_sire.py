"""
UmaEdge — Backfill Broodmare Sire (母父) from Umanity.jp English Pages.

Extracts the dam's sire from the Umanity horse profile page. Uses the English
version so sire names match the existing sire_name format in the horses table.

The pedigree section link order on Umanity EN is:
  1. ">> Detail" (pedigree page link)
  2. Sire link (horse_top.php)
  3. "Pedigree" for sire
  4. "Progeny" for sire
  5. Sire's sire link
  6. Sire's dam link
  7. Dam link (horse_top.php)
  8. "Pedigree" for dam
  9. "Son/Daughter" for dam
  10. **Dam's sire / BMS** (horse_top.php)  ← target
  11. Dam's dam link

Usage:
    python -m scraper.backfill_broodmare_sire                # backfill all
    python -m scraper.backfill_broodmare_sire --limit 100    # first 100
    python -m scraper.backfill_broodmare_sire --dry-run      # preview only
    python -m scraper.backfill_broodmare_sire --workers 4    # concurrency
"""

import argparse
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

from scraper.db import get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_bms")

HORSE_URL = "https://umanity.jp/en/racedata/db/horse_top.php?code={code}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Thread-safe lock for DB writes
_db_lock = threading.Lock()

# Cache for sire name → horse ID lookups
_sire_lookup: dict[str, int] = {}
_sire_lookup_loaded = False

# Skip tokens — links with these texts are navigation, not horse names
_SKIP_NAMES = {"Pedigree", "Progeny", "Son/Daughter", ">> Detail", "Detail"}


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


def parse_broodmare_sire(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract broodmare sire (dam's sire) name from Umanity EN horse page.

    After the ">> Detail" link, horse_top.php links appear in order:
      [0] Sire
      [1] Sire's sire
      [2] Sire's dam
      [3] Dam
      [4] **Dam's sire (BMS)** ← target
      [5] Dam's dam

    We skip navigation links (Pedigree, Progeny, Son/Daughter, Detail).
    """
    # Find the ">> Detail" pedigree link
    detail_link = soup.find("a", string=re.compile(r"Detail"))
    if not detail_link:
        detail_link = soup.find("a", href=re.compile(r"horse_pedigree\.php"))
    if not detail_link:
        return None

    # Walk forward through links, collecting horse_top.php links
    horse_links = []
    elem = detail_link
    for _ in range(30):   # scan up to 30 links ahead
        elem = elem.find_next("a")
        if not elem:
            break
        href = elem.get("href", "")
        name = elem.get_text(strip=True)

        # Stop if we've left the pedigree section
        if "race_21.php" in href or "database_" in href or "race_5.php" in href:
            break

        if "horse_top.php" in href and "code=" in href:
            if name and name not in _SKIP_NAMES:
                horse_links.append(name)

    # horse_links should be: [Sire, Sire's sire, Sire's dam, Dam, BMS, Dam's dam]
    # Index 4 = broodmare sire
    if len(horse_links) >= 5:
        return horse_links[4]

    return None


def _load_sire_lookup():
    """One-time load of all sire_name → horse.id mappings."""
    global _sire_lookup, _sire_lookup_loaded
    if _sire_lookup_loaded:
        return
    try:
        with get_session() as session:
            rows = session.execute(text(
                "SELECT id, sire_name FROM horses WHERE sire_name IS NOT NULL"
            )).fetchall()
            for hid, sname in rows:
                if sname not in _sire_lookup:
                    _sire_lookup[sname] = hid
        log.info(f"Loaded {len(_sire_lookup)} sire name → ID mappings")
    except Exception as e:
        log.error(f"Failed to load sire lookup: {e}")
    _sire_lookup_loaded = True


def _find_or_create_sire_horse(sire_name: str) -> Optional[int]:
    """Find the horse ID for a sire by name, or create a placeholder."""
    if sire_name in _sire_lookup:
        return _sire_lookup[sire_name]

    # Try DB lookup by name
    try:
        with get_session() as session:
            row = session.execute(
                text("SELECT id FROM horses WHERE name = :n OR name_jp = :n ORDER BY id LIMIT 1"),
                {"n": sire_name},
            ).fetchone()
            if row:
                _sire_lookup[sire_name] = row[0]
                return row[0]
    except Exception:
        pass

    # Create placeholder
    try:
        with get_session() as session:
            result = session.execute(
                text(
                    "INSERT INTO horses (name, name_jp, sire_name, netkeiba_id) "
                    "VALUES (:name, :name, :name, :nk) RETURNING id"
                ),
                {"name": sire_name, "nk": f"bms_{sire_name[:40]}"},
            )
            new_id = result.fetchone()[0]
            _sire_lookup[sire_name] = new_id
            log.debug(f"  Created placeholder for BMS: {sire_name} (id={new_id})")
            return new_id
    except Exception as e:
        log.warning(f"  Failed to create placeholder for {sire_name}: {e}")
        return None


def get_horses_missing_bms(limit: Optional[int] = None) -> list[dict]:
    """Get horses missing broodmare_sire_id with valid netkeiba IDs."""
    query = """
        SELECT id, netkeiba_id, name_jp
        FROM horses
        WHERE broodmare_sire_id IS NULL
          AND netkeiba_id IS NOT NULL
          AND netkeiba_id NOT LIKE 'unknown_%'
          AND netkeiba_id NOT LIKE 'jra_%'
          AND netkeiba_id NOT LIKE 'bms_%'
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

    bms_name = parse_broodmare_sire(soup)
    if not bms_name:
        log.debug(f"  [{idx}/{total}] ⏭ No BMS found for {horse['name_jp']}")
        return "skipped"

    if dry_run:
        log.info(f"  [{idx}/{total}] 🔍 {horse['name_jp']} → BMS: {bms_name} (dry run)")
        return "updated"

    with _db_lock:
        bms_id = _find_or_create_sire_horse(bms_name)

    if not bms_id:
        log.warning(f"  [{idx}/{total}] ❌ Could not resolve BMS ID for {bms_name}")
        return "failed"

    with _db_lock:
        with get_session() as session:
            session.execute(
                text("UPDATE horses SET broodmare_sire_id = :bms WHERE id = :hid"),
                {"bms": bms_id, "hid": horse["id"]},
            )
    log.debug(f"  [{idx}/{total}] ✅ {horse['name_jp']} → BMS: {bms_name} (id={bms_id})")
    return "updated"


def backfill_broodmare_sires(limit: Optional[int] = None, dry_run: bool = False,
                              workers: int = 4):
    """Main backfill function with thread-pool concurrency."""
    _load_sire_lookup()

    horses = get_horses_missing_bms(limit)
    total = len(horses)
    log.info(f"Found {total} horses missing broodmare_sire_id (using {workers} workers)")

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
    log.info(f"Backfill complete: {updated} updated, {failed} failed, {skipped} no BMS found")
    log.info(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(
        description="UmaEdge — Backfill broodmare sire (母父) from Umanity.jp"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max horses to process")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, don't update DB")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent workers (default: 4)")
    args = parser.parse_args()
    backfill_broodmare_sires(limit=args.limit, dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()
