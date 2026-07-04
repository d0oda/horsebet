#!/usr/bin/env python3
"""
UmaEdge — Concurrent Batch Scraper.

Implements macro-level concurrency by slicing a year into 12 months.
Each thread handles a single month, finding all race days and scraping the races.

Usage:
    # Scrape 2024 using 4 concurrent threads (months)
    python -m scraper.batch_scrape_concurrent --year 2024 --workers 4

    # Dry run — check what dates/months would be processed
    python -m scraper.batch_scrape_concurrent --year 2024 --dry-run
"""

import argparse
import logging
import time
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from sqlalchemy import text
from scraper.db import get_session
from scraper.netkeiba import scrape_race_list, scrape_race

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Thread-%(threadName)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_concurrent")

def get_already_scraped_ids() -> set[str]:
    """Return the set of netkeiba race IDs already in the database."""
    with get_session() as session:
        result = session.execute(text("SELECT netkeiba_id FROM races"))
        return {row[0] for row in result.fetchall()}

def generate_dates_for_month(year: int, month: int) -> list[str]:
    """Generate all weekend dates (Sat/Sun) for a given month, as these are the primary JRA race days."""
    dates = []
    # Start at the first day of the month
    current_date = date(year, month, 1)
    
    while current_date.month == month:
        # 5 = Saturday, 6 = Sunday
        # Also include Mondays (0) occasionally for holiday racing
        if current_date.weekday() in (0, 5, 6):
            dates.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    return dates

def scrape_date(d: str, already_scraped: set[str], dry_run: bool = False) -> dict:
    """Worker function to scrape all JRA races for a specific date."""
    stats = {
        "date": d,
        "races_found": 0,
        "already_scraped": 0,
        "races_scraped": 0,
        "races_failed": 0,
    }
    
    if dry_run:
        return stats
        
    try:
        # 1. Discover races for this date
        race_ids = scrape_race_list(d)
        if not race_ids:
            time.sleep(0.5)
            return stats
            
        stats["races_found"] += len(race_ids)
        
        # 2. Scrape each race
        for rid in race_ids:
            if rid in already_scraped:
                stats["already_scraped"] += 1
                continue
                
            log.info(f"Date {d}: Scraping race {rid}...")
            result = scrape_race(rid)
            
            if result:
                stats["races_scraped"] += 1
            else:
                stats["races_failed"] += 1
            
            # Polite delay to prevent immediate bans within the thread
            time.sleep(1.5)
            
    except Exception as e:
        log.error(f"Failed processing date {d}: {e}")
        stats["races_failed"] += 1
        time.sleep(5) # Backoff on error
            
    return stats

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Concurrent Batch Scraper")
    parser.add_argument("--year", type=int, required=True, help="Year to scrape (e.g. 2024)")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent threads to run. Max recommended is 4 to avoid IP bans.")
    parser.add_argument("--dry-run", action="store_true", help="Only show dates, don't scrape")

    args = parser.parse_args()
    
    already_scraped = get_already_scraped_ids()
    log.info(f"Found {len(already_scraped)} races already in the database.")
    
    log.info(f"Starting concurrent scrape for year {args.year} with {args.workers} workers.")
    
    # Generate all dates for the year
    all_dates = []
    for month in range(1, 13):
        all_dates.extend(generate_dates_for_month(args.year, month))
    
    if args.dry_run:
        log.info(f"[DRY RUN] Would scan {len(all_dates)} potential race dates: {all_dates[0]} ... {all_dates[-1]}")
        return

    total_stats = {
        "races_found": 0,
        "already_scraped": 0,
        "races_scraped": 0,
        "races_failed": 0,
    }
    
    log.info(f"Queueing {len(all_dates)} days to be processed by {args.workers} workers...")
    
    # Execute across thread pool
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="ScraperWorker") as executor:
        # Submit a future for each specific date
        futures = {
            executor.submit(scrape_date, d, already_scraped, args.dry_run): d 
            for d in all_dates
        }
        
        for future in as_completed(futures):
            d = futures[future]
            try:
                stats = future.result()
                if stats["races_found"] > 0:
                    log.info(f"✅ Date {d} completed: {stats['races_scraped']} scraped, {stats['already_scraped']} skipped.")
                
                total_stats["races_found"] += stats["races_found"]
                total_stats["already_scraped"] += stats["already_scraped"]
                total_stats["races_scraped"] += stats["races_scraped"]
                total_stats["races_failed"] += stats["races_failed"]
            except Exception as exc:
                log.error(f"❌ Date {d} generated an exception: {exc}")

    # Print summary
    print("\n" + "=" * 50)
    print(f"  UmaEdge — Concurrent Scrape Summary ({args.year})")
    print("=" * 50)
    print(f"  Races found:         {total_stats['races_found']:>6}")
    print(f"  Already scraped:     {total_stats['already_scraped']:>6}")
    print(f"  Races scraped:       {total_stats['races_scraped']:>6}")
    print(f"  Failed:              {total_stats['races_failed']:>6}")
    print("=" * 50)

if __name__ == "__main__":
    main()
