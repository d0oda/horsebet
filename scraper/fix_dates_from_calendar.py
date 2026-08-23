"""
UmaEdge — Fix Race Dates from JRA Calendar.

Scrapes the official JRA race fixtures from japanracing.jp to get actual
race dates, then maps netkeiba race IDs to calendar dates.

The netkeiba race ID format is YYYYCCDDRRNN:
  YYYY = year, CC = course (venue), DD = meeting, RR = round, NN = race number

The key insight: for each venue in a year, meetings and rounds are assigned
sequentially in calendar order. So by counting the race days per venue from
the JRA calendar, we can map (CC, DD, RR) -> actual date.

Usage:
    python -m scraper.fix_dates_from_calendar                # fix all years  
    python -m scraper.fix_dates_from_calendar --year 2025    # fix one year
    python -m scraper.fix_dates_from_calendar --dry-run      # preview only
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scraper.db import get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fix_dates_calendar")

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# Map JRA venue image filename prefixes to netkeiba course codes
IMG_TO_VENUE = {
    "rf_sap": "01",  # Sapporo
    "rf_hak": "02",  # Hakodate
    "rf_fuk": "03",  # Fukushima
    "rf_nii": "04",  # Niigata
    "rf_tok": "05",  # Tokyo
    "rf_nak": "06",  # Nakayama
    "rf_chu": "07",  # Chukyo
    "rf_kyo": "08",  # Kyoto
    "rf_han": "09",  # Hanshin
    "rf_kok": "10",  # Kokura
}

VENUE_NAMES = {
    "01": "Sapporo", "02": "Hakodate", "03": "Fukushima", "04": "Niigata",
    "05": "Tokyo", "06": "Nakayama", "07": "Chukyo", "08": "Kyoto",
    "09": "Hanshin", "10": "Kokura",
}

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def scrape_jra_calendar(year: int) -> dict:
    """Scrape the JRA race calendar for a given year from japanracing.jp.
    
    Returns:
        dict: {date_str: [venue_code, ...]} for each race day
    """
    url = f"https://japanracing.jp/en/racing/schedule/jra/{year}.html"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        log.error(f"Failed to fetch JRA calendar for {year}: HTTP {resp.status_code}")
        return {}
    
    soup = BeautifulSoup(resp.text, "lxml")
    race_days = {}
    
    for table in soup.find_all("table"):
        # Determine which month this table belongs to
        prev = table.find_previous(string=re.compile("|".join(MONTH_NAMES)))
        if not prev:
            continue
        month_idx = None
        for i, m in enumerate(MONTH_NAMES):
            if m in prev.strip():
                month_idx = i + 1
                break
        if not month_idx:
            continue
        
        for td in table.find_all("td"):
            day_span = td.find("span", class_="dayNumber")
            if not day_span:
                continue
            day_text = day_span.get_text(strip=True)
            if not day_text or not day_text.isdigit():
                continue
            day = int(day_text)
            
            # Find venue icons
            icons = td.find_all("img", src=re.compile(r"rf_\w+"))
            venues = set()
            for img in icons:
                src = img.get("src", "")
                for key, code in IMG_TO_VENUE.items():
                    if key in src:
                        venues.add(code)
            
            if venues:
                date_str = f"{year}-{month_idx:02d}-{day:02d}"
                race_days[date_str] = sorted(list(venues))
    
    return race_days


def build_venue_day_sequence(calendar: dict) -> dict:
    """Build a per-venue chronological sequence of race days.
    
    For each venue, returns a list of dates in chronological order.
    This allows mapping sequential (meeting, round) numbers to actual dates.
    
    Returns:
        dict: {venue_code: [date1, date2, ...]}
    """
    venue_days = defaultdict(list)
    for date_str in sorted(calendar.keys()):
        for venue in calendar[date_str]:
            venue_days[venue].append(date_str)
    return dict(venue_days)


def compute_meeting_round_mapping(venue_days: dict) -> dict:
    """Compute the mapping from (venue, meeting, round) -> actual_date.
    
    JRA meetings at each venue have up to 6 rounds (race days).
    Meetings are numbered sequentially, and rounds within each meeting
    are consecutive days at that venue.
    
    Returns:
        dict: {(venue_code, meeting_str, round_str): date_str}
    """
    mapping = {}
    
    for venue, days in venue_days.items():
        meeting = 1
        round_num = 1
        
        for date_str in days:
            meeting_str = f"{meeting:02d}"
            round_str = f"{round_num:02d}"
            mapping[(venue, meeting_str, round_str)] = date_str
            
            round_num += 1
            if round_num > 6:
                # New meeting
                meeting += 1
                round_num = 1
    
    return mapping


def get_bad_date_combos(year: int = None) -> list[dict]:
    """Get all unique (netkeiba_id prefix) combos with bad dates from the DB."""
    with get_session() as session:
        query = """
            SELECT 
                SUBSTRING(netkeiba_id FROM 1 FOR 10) as prefix,
                SUBSTRING(netkeiba_id FROM 5 FOR 2) as course,
                SUBSTRING(netkeiba_id FROM 7 FOR 2) as meeting,
                SUBSTRING(netkeiba_id FROM 9 FOR 2) as round_num,
                COUNT(*) as race_count,
                CAST(date AS TEXT) as bad_date
            FROM races
            WHERE date IN (
                SELECT date FROM races
                GROUP BY date HAVING COUNT(*) > 36
            )
        """
        params = {}
        if year:
            query += " AND netkeiba_id LIKE :year_prefix"
            params["year_prefix"] = f"{year}%"
        query += """
            GROUP BY prefix, course, meeting, round_num, date
            ORDER BY prefix
        """
        rows = session.execute(text(query), params).fetchall()
    
    return [
        {
            "prefix": r[0],
            "course": r[1],
            "meeting": r[2],
            "round": r[3],
            "count": r[4],
            "bad_date": r[5],
        }
        for r in rows
    ]


def fix_dates(year: int, dry_run: bool = False):
    """Fix race dates for a specific year."""
    log.info(f"Fixing race dates for {year}...")
    
    # 1. Scrape calendar
    calendar = scrape_jra_calendar(year)
    if not calendar:
        log.error(f"No calendar data for {year}")
        return 0, 0
    log.info(f"  Calendar: {len(calendar)} race days found")
    
    # 2. Build venue sequences and mapping
    venue_days = build_venue_day_sequence(calendar)
    for venue, days in sorted(venue_days.items()):
        log.info(f"  {VENUE_NAMES.get(venue, venue)}: {len(days)} race days")
    
    mapping = compute_meeting_round_mapping(venue_days)
    log.info(f"  Total (venue, meeting, round) combos in calendar: {len(mapping)}")
    
    # 3. Get bad combos from DB
    combos = get_bad_date_combos(year)
    log.info(f"  Bad date combos in DB: {len(combos)}")
    
    if not combos:
        log.info(f"  Nothing to fix for {year}!")
        return 0, 0
    
    # 4. Match and fix
    fixed = 0
    not_found = 0
    
    for combo in combos:
        key = (combo["course"], combo["meeting"], combo["round"])
        actual_date = mapping.get(key)
        
        if not actual_date:
            not_found += 1
            if not_found <= 10:
                log.warning(
                    f"  ❌ No calendar match for {combo['prefix']} "
                    f"(venue={combo['course']} meeting={combo['meeting']} round={combo['round']})"
                )
            continue
        
        if actual_date == combo["bad_date"]:
            continue  # Already correct
        
        fixed += 1
        if fixed <= 50 or fixed % 50 == 0:
            log.info(
                f"  ✅ {combo['prefix']}: {combo['bad_date']} → {actual_date} "
                f"({combo['count']} races, {VENUE_NAMES.get(combo['course'], '?')})"
            )
        
        if not dry_run:
            with get_session() as session:
                session.execute(
                    text("""
                        UPDATE races 
                        SET date = :new_date 
                        WHERE SUBSTRING(netkeiba_id FROM 1 FOR 10) = :prefix
                    """),
                    {"new_date": actual_date, "prefix": combo["prefix"]},
                )
    
    return fixed, not_found


def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Fix Race Dates from JRA Calendar")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no DB changes")
    parser.add_argument("--year", type=int, help="Fix only specific year")
    args = parser.parse_args()
    
    if args.year:
        years = [args.year]
    else:
        # Fix all affected years
        years = [2019, 2020, 2022, 2023, 2024, 2025]
    
    total_fixed = 0
    total_not_found = 0
    
    for year in years:
        fixed, not_found = fix_dates(year, args.dry_run)
        total_fixed += fixed
        total_not_found += not_found
        log.info(f"\n  {year}: fixed={fixed}, not_found={not_found}\n")
    
    log.info(f"\n{'=' * 60}")
    log.info(f"  Date Fix Summary")
    log.info(f"{'=' * 60}")
    log.info(f"  Years processed: {years}")
    log.info(f"  Total fixed:     {total_fixed}")
    log.info(f"  Not found:       {total_not_found}")
    if args.dry_run:
        log.info(f"  (dry run — no changes made)")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
