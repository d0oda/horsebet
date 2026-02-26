"""
Backfill sire_name for existing horses by scraping their netkeiba profile pages.

Usage:
    python -m scraper.backfill_sires                # backfill all missing
    python -m scraper.backfill_sires --limit 100    # backfill first 100
    python -m scraper.backfill_sires --dry-run      # parse only, don't update DB
"""

import argparse
import logging
import random
import re
import time
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


def _fetch(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch a URL with retries and polite delay."""
    for attempt in range(retries):
        try:
            delay = random.uniform(1.5, 3.5)
            time.sleep(delay)
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.encoding = "EUC-JP"
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            elif resp.status_code == 404:
                log.debug(f"404: {url}")
                return None
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


def backfill_sires(limit: Optional[int] = None, dry_run: bool = False):
    """Main backfill function."""
    horses = get_horses_missing_sire(limit)
    log.info(f"Found {len(horses)} horses missing sire_name")

    if not horses:
        log.info("Nothing to backfill!")
        return

    updated = 0
    failed = 0
    skipped = 0

    for i, horse in enumerate(horses, 1):
        nk_id = horse["netkeiba_id"]
        url = HORSE_PED_URL.format(horse_nk_id=nk_id)

        soup = _fetch(url)
        if not soup:
            failed += 1
            log.warning(f"  [{i}/{len(horses)}] ❌ Failed to fetch {horse['name_jp']} ({nk_id})")
            continue

        sire_name = parse_sire_name(soup)
        if not sire_name:
            skipped += 1
            log.debug(f"  [{i}/{len(horses)}] ⏭ No sire found for {horse['name_jp']} ({nk_id})")
            continue

        if dry_run:
            log.info(f"  [{i}/{len(horses)}] 🔍 {horse['name_jp']} → sire: {sire_name} (dry run)")
            updated += 1
            continue

        # Update DB
        with get_session() as session:
            session.execute(
                text("UPDATE horses SET sire_name = :sire WHERE id = :hid"),
                {"sire": sire_name, "hid": horse["id"]},
            )
        updated += 1
        log.info(f"  [{i}/{len(horses)}] ✅ {horse['name_jp']} → sire: {sire_name}")

    log.info(f"\n{'=' * 50}")
    log.info(f"Backfill complete: {updated} updated, {failed} failed, {skipped} no sire found")
    log.info(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Backfill sire names from netkeiba horse profiles")
    parser.add_argument("--limit", type=int, default=None, help="Max horses to process")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, don't update DB")
    args = parser.parse_args()
    backfill_sires(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
