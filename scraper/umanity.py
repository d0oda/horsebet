"""
UmaEdge — Umanity.jp Race Scraper.

Scrapes JRA race cards from umanity.jp as an alternative to netkeiba
(which rate-limits aggressively).

Produces the same RaceData/EntryData/HorseData objects used by the rest
of the pipeline, so `save_race_to_db()` works unchanged.

Usage:
    # Scrape all races for a date
    python -m scraper.umanity --date 2026-03-07

    # Scrape a single race by Umanity code
    python -m scraper.umanity --code 2026030706020301

    # Dry-run (list discovered races without scraping)
    python -m scraper.umanity --date 2026-03-07 --dry-run
"""

import argparse
import logging
import os
import random
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

from scraper.db import get_session
from scraper.netkeiba import (
    RaceData, EntryData, HorseData,
    save_race_to_db, _extract_race_class,
    JRA_COURSES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("umanity")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://umanity.jp"
# race_8_1.php is the 出馬表 sub-tab (has entry table inline)
RACE_CARD_URL = BASE_URL + "/racedata/race_8_1.php?code={code}"
PEDIGREE_URL = BASE_URL + "/racedata/race_8_4.php?code={code}"
RACE_MAIN_URL = BASE_URL + "/racedata/race_8.php?code={code}"
PROGRAM_URL = BASE_URL + "/racedata/race_5.php?date={date}"

DELAY_MIN = float(os.getenv("UMANITY_DELAY_MIN", 0.5))
DELAY_MAX = float(os.getenv("UMANITY_DELAY_MAX", 1.5))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sleep():
    """Random delay between requests."""
    delay = random.uniform(DELAY_MIN, DELAY_MAX)
    time.sleep(delay)


def _fetch(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch a URL and return parsed BeautifulSoup."""
    for attempt in range(retries):
        try:
            _sleep()
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.encoding = "utf-8"
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            elif resp.status_code == 404:
                log.warning(f"404 Not Found: {url}")
                return None
            else:
                log.warning(f"HTTP {resp.status_code} for {url} (attempt {attempt + 1})")
        except requests.RequestException as e:
            log.warning(f"Request error for {url}: {e} (attempt {attempt + 1})")

        if attempt < retries - 1:
            time.sleep(3 * (attempt + 1))

    log.error(f"Failed to fetch {url} after {retries} attempts")
    return None


def umanity_code_to_netkeiba_id(code: str) -> str:
    """
    Convert a 16-digit Umanity race code to a 12-digit netkeiba race ID.

    Umanity: YYYYMMDDVVKKDDRR  (date[8] + venue[2] + kai[2] + day[2] + race[2])
    Example: 2026030706020311 → 202606020311

    Netkeiba: YYYYVVKKDDRR (12 digits)
    """
    year = code[:4]
    venue_onwards = code[8:16]  # VVKKDDRR
    return year + venue_onwards


def _extract_umanity_horse_code(href: str) -> Optional[str]:
    """Extract horse code from a Umanity URL like 'horse_top.php?code=2023105263'."""
    match = re.search(r'code=(\d+)', href)
    return match.group(1) if match else None


def _normalize_digits(text: str) -> str:
    """Convert full-width digits to half-width."""
    return text.translate(str.maketrans('０１２３４５６７８９', '0123456789'))


# ---------------------------------------------------------------------------
# Parsing — Race Header
# ---------------------------------------------------------------------------

def _parse_race_header(soup: BeautifulSoup, code: str) -> RaceData:
    """Parse race header from a Umanity race card page."""
    netkeiba_id = umanity_code_to_netkeiba_id(code)
    race = RaceData(netkeiba_id=netkeiba_id)

    # Gather all text from the page to search for race details.
    # Umanity layouts vary, so we search broadly.
    page_text = soup.get_text(strip=False)

    # Race name from <title>: "3歳未勝利 - 2026年3月7日中山5R｜出馬表｜..."
    title = soup.find("title")
    if title:
        title_text = title.get_text(strip=True)
        name_match = re.match(r'(.+?)\s*-\s*\d{4}年', title_text)
        if name_match:
            race.race_name_jp = name_match.group(1).strip()
            race.race_name = race.race_name_jp

    # Date from page
    date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', page_text)
    if date_match:
        race.date = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"
    elif len(code) >= 8:
        race.date = f"{code[:4]}-{code[4:6]}-{code[6:8]}"

    # Post time: "HH:MM発走"
    time_match = re.search(r'(\d{1,2}:\d{2})発走', page_text)
    if time_match:
        race.post_time = time_match.group(1)
    else:
        time_match = re.search(r'(\d{1,2}:\d{2})', page_text[:500])
        if time_match:
            race.post_time = time_match.group(1)

    # Surface and distance: "ダート・右 1200m" or "芝・右外 2200m" or "障3200m"
    dist_match = re.search(r'(ダート|芝|障)[^\d]{0,10}(\d{3,4})m', page_text)
    if dist_match:
        surface_str = dist_match.group(1)
        race.distance = int(dist_match.group(2))
        race.surface = 'dirt' if surface_str == 'ダート' else 'turf'
    else:
        dist_match = re.search(r'(\d{3,4})m', page_text[:500])
        if dist_match:
            race.distance = int(dist_match.group(1))
        if 'ダート' in page_text[:500]:
            race.surface = 'dirt'
        elif '芝' in page_text[:500]:
            race.surface = 'turf'

    # Class from race name or page text
    norm_name = _normalize_digits(race.race_name_jp or '')
    race.class_ = _extract_race_class(norm_name, None)

    # Also try the text after the last "｜" in the race info area
    if not race.class_:
        class_match = re.search(r'[｜|]\s*([^｜|\n]+?)\s*(?:[指定]|馬齢|$)', page_text[:500])
        if class_match:
            norm = _normalize_digits(class_match.group(1).strip())
            race.class_ = _extract_race_class(norm, None)

    # Course code and race number from the netkeiba ID
    if len(netkeiba_id) >= 12:
        race.course_code = netkeiba_id[4:6]
        try:
            race.race_number = int(netkeiba_id[10:12])
        except (ValueError, IndexError):
            pass

    # Grade detection
    name_text = race.race_name_jp or ''
    for g in ['G1', 'G2', 'G3', 'GI', 'GII', 'GIII', 'Ｇ１', 'Ｇ２', 'Ｇ３']:
        if g in name_text:
            race.grade = g.replace('Ｇ', 'G').replace('I', '1').replace('II', '2').replace('III', '3')
            break

    if race.grade and not race.class_:
        race.class_ = race.grade

    return race


# ---------------------------------------------------------------------------
# Parsing — Entry Table
# ---------------------------------------------------------------------------

def _parse_entries(soup: BeautifulSoup, race: RaceData) -> list[EntryData]:
    """
    Parse entries from the Umanity race card table (race_8_1.php).

    Each data row has fixed cell layout (verified empirically):
        Cell 0:  枠番 (draw) — "-" before Friday
        Cell 1:  馬番 (post pos) — "-" before Friday
        Cell 2:  My予想印 (skip)
        Cell 3:  U指数 (skip)
        Cell 4:  馬名 (short, with horse_top link)
        Cell 5:  (empty / image)
        Cell 6:  馬名 (full, with horse_top link)
        Cell 7:  性齢 (sex+age)
        Cell 8:  斤量 (weight carried)
        Cell 9:  騎手 (jockey, with database_jockey link)
        Cell 10: 調教師 (trainer, with database_trainer link)
        Cell 11: 所属 (affiliation)
        Cell 12: ブリンカー
        Cell 13: みんなの人気
        Cell 14+: 近走成績 (recent results)
    """
    entries = []

    # Find rows with horse links across all tables
    entry_rows = []
    seen_horse_codes = set()

    for table in soup.find_all('table'):
        for row in table.find_all('tr'):
            horse_links = row.select("a[href*='horse_top.php']")
            if not horse_links:
                continue

            cells = row.find_all('td')
            # Valid entry rows have 11-30 cells. Skip mega-rows (flat table)
            # where all entries appear in a single <tr> with 300+ cells.
            if len(cells) < 11 or len(cells) > 30:
                continue

            # Get horse code to deduplicate
            horse_code = _extract_umanity_horse_code(horse_links[0].get('href', ''))
            if horse_code and horse_code in seen_horse_codes:
                continue
            if horse_code:
                seen_horse_codes.add(horse_code)

            entry_rows.append(row)

    if not entry_rows:
        log.warning(f"No entry rows found for {race.netkeiba_id}")
        return entries

    for idx, row in enumerate(entry_rows, 1):
        cells = row.find_all('td')

        try:
            # Draw (枠番) - cell 0
            draw_text = cells[0].get_text(strip=True)
            draw = int(draw_text) if draw_text.isdigit() else None

            # Post position (馬番) - cell 1
            pp_text = cells[1].get_text(strip=True)
            post_pos = int(pp_text) if pp_text.isdigit() else idx  # fallback to sequential

            # Horse name + code - find via link
            horse_link = row.select_one("a[href*='horse_top.php']")
            horse_name_jp = horse_link.get_text(strip=True) if horse_link else '?'
            horse_id = _extract_umanity_horse_code(horse_link.get('href', '')) if horse_link else None

            # Sex and age - cell 7
            sex_age = _normalize_digits(cells[7].get_text(strip=True))
            sex = sex_age[0] if sex_age else None
            birth_year = None
            if sex_age and len(sex_age) >= 2:
                try:
                    age = int(sex_age[1:])
                    if race.date:
                        birth_year = int(race.date[:4]) - age
                except ValueError:
                    pass

            # Weight carried (斤量) - cell 8
            weight_carried = None
            wt_text = cells[8].get_text(strip=True)
            wt_text = re.sub(r'^[▲△☆★]', '', wt_text)
            try:
                weight_carried = float(wt_text)
            except ValueError:
                pass

            # Jockey - cell 9 (or find via link)
            jockey_name_jp = None
            jockey_link = row.select_one("a[href*='database_jockey']")
            if jockey_link:
                jockey_name_jp = jockey_link.get_text(strip=True)
            elif len(cells) > 9:
                jockey_name_jp = cells[9].get_text(strip=True)

            # Trainer - cell 10 (or find via link)
            trainer_name_jp = None
            trainer_link = row.select_one("a[href*='database_trainer']")
            if trainer_link:
                trainer_name_jp = trainer_link.get_text(strip=True)
            elif len(cells) > 10:
                trainer_name_jp = cells[10].get_text(strip=True)

            horse = HorseData(
                name=horse_name_jp,
                name_jp=horse_name_jp,
                sex=sex,
                birth_year=birth_year,
                netkeiba_id=horse_id or f"uma_{entry_idx}",
            )

            entry = EntryData(
                post_position=post_pos,
                draw=draw,
                horse=horse,
                jockey_name_jp=jockey_name_jp,
                trainer_name_jp=trainer_name_jp,
                weight_carried=weight_carried,
            )
            entries.append(entry)

        except Exception as e:
            log.warning(f"Error parsing entry row {entry_idx} in {race.netkeiba_id}: {e}")
            continue

    return entries


# ---------------------------------------------------------------------------
# Parsing — Pedigree
# ---------------------------------------------------------------------------

def _fetch_pedigree(code: str, entries: list[EntryData]):
    """Fetch pedigree tab (race_8_4.php) and update entries with sire/dam."""
    url = PEDIGREE_URL.format(code=code)
    soup = _fetch(url)
    if not soup:
        return

    # Find the entry table on the pedigree page (same structure as race card)
    # Rows with horse links correspond to our entries in order
    pedigree_rows = []
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            if row.select("a[href*='horse_top.php']"):
                pedigree_rows.append(row)

    entry_idx = 0
    seen_horses = set()
    for row in pedigree_rows:
        if entry_idx >= len(entries):
            break

        # Skip duplicate rows for the same horse
        horse_link = row.select_one("a[href*='horse_top.php']")
        if horse_link:
            horse_code = _extract_umanity_horse_code(horse_link.get("href", ""))
            if horse_code in seen_horses:
                continue
            seen_horses.add(horse_code)

        # Find pedigree links (horse_pedigree.php)
        ped_links = row.select("a[href*='horse_pedigree.php']")
        if len(ped_links) >= 1:
            entries[entry_idx].horse.sire_name = ped_links[0].get_text(strip=True)
        if len(ped_links) >= 2:
            entries[entry_idx].horse.dam_name = ped_links[1].get_text(strip=True)

        entry_idx += 1


# ---------------------------------------------------------------------------
# Main Scraping Functions
# ---------------------------------------------------------------------------

def scrape_umanity_race(code: str, fetch_pedigree: bool = True) -> Optional[RaceData]:
    """
    Scrape a single race from Umanity.

    Args:
        code: 16-digit Umanity race code (e.g., '2026030706020301')
        fetch_pedigree: Whether to also fetch sire/dam info (extra HTTP request)

    Returns:
        RaceData object or None if failed/already exists.
    """
    netkeiba_id = umanity_code_to_netkeiba_id(code)

    # Check if already in DB
    with get_session() as session:
        existing = session.execute(
            text("SELECT id FROM races WHERE netkeiba_id = :nid"),
            {"nid": netkeiba_id},
        ).fetchone()
        if existing:
            log.info(f"Race {netkeiba_id} already in DB (id={existing[0]}), skipping")
            return None

    # Fetch race card page (race_8_1.php has inline entry table)
    url = RACE_CARD_URL.format(code=code)
    soup = _fetch(url)
    if not soup:
        return None

    # Parse header (primary: from race_8_1.php)
    race = _parse_race_header(soup, code)
    if not race.race_name_jp:
        log.warning(f"No race header found for {code}")
        return None

    # If surface/distance not found, fetch from race_8.php (parent page)
    if not race.surface or not race.distance:
        main_url = RACE_MAIN_URL.format(code=code)
        main_soup = _fetch(main_url)
        if main_soup:
            main_text = main_soup.get_text(strip=False)[:1500]
            if not race.surface:
                dist_match = re.search(r'(ダート|芝|障)[^\d]{0,10}(\d{3,4})m', main_text)
                if dist_match:
                    race.surface = 'dirt' if dist_match.group(1) == 'ダート' else 'turf'
                    race.distance = int(dist_match.group(2))
                else:
                    if 'ダート' in main_text:
                        race.surface = 'dirt'
                    elif '芝' in main_text:
                        race.surface = 'turf'
            if not race.distance:
                dist_match = re.search(r'(\d{3,4})m', main_text)
                if dist_match:
                    race.distance = int(dist_match.group(1))

    # Parse entries
    entries = _parse_entries(soup, race)
    race.entries = entries
    race.field_size = len(entries)

    if not entries:
        log.warning(f"No entries parsed for {code} ({race.race_name_jp})")
        return None

    # Fetch pedigree info
    if fetch_pedigree and entries:
        _fetch_pedigree(code, entries)

    # Save to DB
    saved = save_race_to_db(race)
    if saved:
        log.info(f"✅ Saved {netkeiba_id} ({race.race_name_jp}, {race.date}) — {len(entries)} entries")

    return race


def scrape_umanity_race_list(date_str: str) -> list[str]:
    """
    Discover all JRA race codes for a date from Umanity's program page.

    Args:
        date_str: Date in YYYYMMDD format

    Returns:
        List of 16-digit Umanity race codes
    """
    date_slash = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}"
    url = PROGRAM_URL.format(date=date_slash)
    log.info(f"Fetching Umanity race program for {date_str}...")
    soup = _fetch(url)
    if not soup:
        return []

    codes = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        match = re.search(r'(?:race_8|race_21)\.php\?code=(\d{16})', href)
        if match:
            code = match.group(1)
            venue_code = code[8:10]
            try:
                if int(venue_code) <= 10 and code not in codes and code.startswith(date_str):
                    codes.append(code)
            except ValueError:
                continue

    log.info(f"Found {len(codes)} JRA races for {date_str}")
    return codes


def scrape_date(date: str, dry_run: bool = False, skip_pedigree: bool = False) -> dict:
    """
    Scrape all JRA races for a given date.

    Args:
        date: Date in YYYY-MM-DD format
        dry_run: If True, just list races without scraping

    Returns:
        Stats dict with counts
    """
    date_compact = date.replace("-", "")
    codes = scrape_umanity_race_list(date_compact)

    if not codes:
        log.error(f"No races found for {date}")
        return {"found": 0, "saved": 0, "skipped": 0, "failed": 0}

    stats = {"found": len(codes), "saved": 0, "skipped": 0, "failed": 0}

    if dry_run:
        for code in codes:
            nk_id = umanity_code_to_netkeiba_id(code)
            log.info(f"  Would scrape: {code} → {nk_id}")
        return stats

    for i, code in enumerate(codes, 1):
        try:
            result = scrape_umanity_race(code, fetch_pedigree=not skip_pedigree)
            if result:
                stats["saved"] += 1
            else:
                stats["skipped"] += 1
        except Exception as e:
            log.warning(f"  [{i}/{len(codes)}] ❌ {code}: {e}")
            stats["failed"] += 1

    log.info(f"\n{'=' * 50}")
    log.info(f"Date {date}: {stats['saved']} saved, {stats['skipped']} skipped, {stats['failed']} failed")
    log.info(f"{'=' * 50}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UmaEdge — Umanity Race Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", help="Race date (YYYY-MM-DD)")
    parser.add_argument("--code", help="Single Umanity race code (16 digits)")
    parser.add_argument("--dry-run", action="store_true", help="List races without scraping")
    parser.add_argument("--skip-pedigree", action="store_true", help="Skip pedigree fetch")
    args = parser.parse_args()

    if args.code:
        result = scrape_umanity_race(args.code, fetch_pedigree=not args.skip_pedigree)
        if result:
            print(f"\nRace: {result.race_name_jp}")
            print(f"Date: {result.date}, Surface: {result.surface}, Distance: {result.distance}m")
            print(f"Entries: {len(result.entries)}")
            for e in result.entries:
                sire = e.horse.sire_name or "?"
                print(f"  #{e.post_position}: {e.horse.name_jp} ({e.horse.sex}{e.horse.birth_year and (int(result.date[:4]) - e.horse.birth_year) or '?'}) "
                      f"J:{e.jockey_name_jp} T:{e.trainer_name_jp} sire:{sire}")
    elif args.date:
        scrape_date(args.date, dry_run=args.dry_run, skip_pedigree=args.skip_pedigree)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
