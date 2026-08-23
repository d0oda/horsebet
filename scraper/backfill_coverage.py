"""
UmaEdge — Backfill Race Coverage by Date.

Scrapes JRA races by iterating through calendar dates (every Sat/Sun + select
weekdays) rather than guessing race IDs. Uses db.netkeiba.com/race/list/
to discover historical race IDs for each date (static HTML, works for past dates).

JRA race days: primarily Sat/Sun year-round, plus some Fridays/Mondays.
~288 race days per year.

Usage:
    python -m scraper.backfill_coverage --year 2021               # full year
    python -m scraper.backfill_coverage --year 2021 --month 6     # June 2021
    python -m scraper.backfill_coverage --start 2020-03-01 --end 2020-12-31
    python -m scraper.backfill_coverage --all-gaps                # auto-detect
    python -m scraper.backfill_coverage --dry-run --year 2021     # preview
"""

import argparse
import logging
import re
import time
from datetime import date, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

from scraper.db import get_session
from scraper.netkeiba import scrape_race, save_race_to_db, HEADERS, _fetch

# Historical race list URL (static HTML, works for past dates)
RACE_LIST_HISTORY_URL = "https://db.netkeiba.com/race/list/{date}/"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_coverage")


def get_jra_race_dates(start: date, end: date) -> list[date]:
    """
    Generate candidate JRA race dates between start and end.
    JRA races on: Sat, Sun, plus some Fri/Mon/Wed holidays.
    We check every day to catch irregular schedules.
    """
    dates = []
    current = start
    while current <= end:
        # JRA races primarily on weekends, but we check all days
        # to catch holiday meetings and irregular schedules.
        # Weekday filtering: Mon=0 ... Sun=6
        # Primary: Sat(5), Sun(6). Secondary: Fri(4), Mon(0).
        # Skip Tue(1), Wed(2), Thu(3) to save time.
        if current.weekday() in (0, 4, 5, 6):  # Mon, Fri, Sat, Sun
            dates.append(current)
        current += timedelta(days=1)
    return dates


def get_dates_already_scraped() -> set[str]:
    """Return set of dates (YYYY-MM-DD strings) already in the database."""
    with get_session() as session:
        rows = session.execute(
            text("SELECT DISTINCT CAST(date AS TEXT) FROM races ORDER BY date")
        ).fetchall()
    return {r[0] for r in rows}


def detect_gaps() -> list[tuple[date, date]]:
    """
    Auto-detect missing date ranges based on current DB coverage.
    Returns list of (start, end) tuples.
    """
    scraped = get_dates_already_scraped()
    gaps = []

    # 2021 is completely missing
    gaps.append((date(2021, 1, 1), date(2021, 12, 31)))

    # For other years, extend to cover full year
    year_ranges = {
        2020: (date(2020, 2, 3), date(2020, 12, 31)),   # has Jan
        2022: (date(2022, 4, 4), date(2022, 12, 31)),   # has Jan-Apr
        2023: (date(2023, 4, 2), date(2023, 12, 31)),   # has Jan-Apr
        2024: (date(2024, 5, 6), date(2024, 12, 31)),   # has Jan-May
        2025: (date(2025, 6, 4), date(2025, 12, 31)),   # has Jan-Jun
    }
    for year, (start, end) in year_ranges.items():
        gaps.append((start, end))

    return gaps


def scrape_date_range(start: date, end: date, dry_run: bool = False) -> dict:
    """Scrape all JRA race dates in a range. Returns stats dict."""
    candidate_dates = get_jra_race_dates(start, end)
    already_scraped = get_dates_already_scraped()

    stats = {"dates_checked": 0, "dates_with_races": 0, "races_saved": 0,
             "dates_skipped": 0, "dates_empty": 0}

    log.info(f"Checking {len(candidate_dates)} candidate dates "
             f"from {start} to {end}")

    for i, d in enumerate(candidate_dates, 1):
        date_str = d.strftime("%Y-%m-%d")
        date_compact = d.strftime("%Y%m%d")

        # Skip if we already have races for this date
        if date_str in already_scraped:
            stats["dates_skipped"] += 1
            continue

        stats["dates_checked"] += 1

        if dry_run:
            log.info(f"  [{i}/{len(candidate_dates)}] Would scrape {date_str}")
            continue

        # Discover races via db.netkeiba.com historical race list
        url = RACE_LIST_HISTORY_URL.format(date=date_compact)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.encoding = "EUC-JP"
            if resp.status_code != 200:
                stats["dates_empty"] += 1
                time.sleep(1)
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            links = soup.find_all("a", href=re.compile(r"/race/\d{12}"))
            race_ids = []
            for link in links:
                m = re.search(r"/race/(\d{12})", link["href"])
                if m and m.group(1) not in race_ids:
                    race_ids.append(m.group(1))
            # Filter to JRA races only (venue codes 01-10)
            # NAR/local racing uses venue codes 30+ and our parser can't handle them
            race_ids = [r for r in race_ids if int(r[4:6]) <= 10]
        except Exception as e:
            log.warning(f"  [{i}/{len(candidate_dates)}] {date_str}: fetch error {e}")
            stats["dates_empty"] += 1
            time.sleep(1)
            continue

        if not race_ids:
            stats["dates_empty"] += 1
            if i <= 20 or i % 50 == 0:
                log.debug(f"  [{i}/{len(candidate_dates)}] {date_str}: no races found")
            time.sleep(0.5)
            continue

        stats["dates_with_races"] += 1
        saved_count = 0

        for rid in race_ids:
            try:
                result = scrape_race(rid)
                if result:
                    saved_count += 1
            except Exception as e:
                log.warning(f"  Failed to scrape {rid}: {e}")
            time.sleep(1)  # rate limit between races

        stats["races_saved"] += saved_count
        log.info(f"  [{i}/{len(candidate_dates)}] {date_str}: "
                 f"saved {saved_count}/{len(race_ids)} races")

        # Progress report every 10 dates with races
        if stats["dates_with_races"] % 10 == 0:
            log.info(f"  Progress: {stats['dates_with_races']} race days, "
                     f"+{stats['races_saved']} races "
                     f"({stats['dates_checked']} checked, "
                     f"{stats['dates_skipped']} skipped)")

        time.sleep(1)  # rate limit between dates

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="UmaEdge — Backfill Race Coverage by Date",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--year", type=int, help="Scrape a full year")
    parser.add_argument("--month", type=int, help="With --year, limit to a specific month")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--all-gaps", action="store_true",
                        help="Auto-detect and fill all coverage gaps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview dates without scraping")
    args = parser.parse_args()

    if args.all_gaps:
        gaps = detect_gaps()
        log.info(f"Detected {len(gaps)} coverage gaps:")
        for s, e in gaps:
            log.info(f"  {s} → {e}")

        total_stats = {"dates_checked": 0, "dates_with_races": 0,
                       "races_saved": 0, "dates_skipped": 0, "dates_empty": 0}
        for s, e in gaps:
            log.info(f"\n{'=' * 50}")
            log.info(f"Filling gap: {s} → {e}")
            log.info(f"{'=' * 50}")
            stats = scrape_date_range(s, e, dry_run=args.dry_run)
            for k in total_stats:
                total_stats[k] += stats[k]

        log.info(f"\n{'=' * 50}")
        log.info(f"All gaps complete:")
        log.info(f"  Race days found: {total_stats['dates_with_races']}")
        log.info(f"  Races saved: {total_stats['races_saved']}")
        log.info(f"  Dates checked: {total_stats['dates_checked']}")
        log.info(f"  Already in DB: {total_stats['dates_skipped']}")
        log.info(f"{'=' * 50}")

    elif args.year:
        if args.month:
            import calendar
            _, last_day = calendar.monthrange(args.year, args.month)
            start = date(args.year, args.month, 1)
            end = date(args.year, args.month, last_day)
        else:
            start = date(args.year, 1, 1)
            end = date(args.year, 12, 31)

        log.info(f"Scraping {start} → {end}")
        stats = scrape_date_range(start, end, dry_run=args.dry_run)

        log.info(f"\n{'=' * 50}")
        log.info(f"Complete: {stats['dates_with_races']} race days, "
                 f"+{stats['races_saved']} races")
        log.info(f"{'=' * 50}")

    elif args.start and args.end:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        log.info(f"Scraping {start} → {end}")
        stats = scrape_date_range(start, end, dry_run=args.dry_run)

        log.info(f"\n{'=' * 50}")
        log.info(f"Complete: {stats['dates_with_races']} race days, "
                 f"+{stats['races_saved']} races")
        log.info(f"{'=' * 50}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
