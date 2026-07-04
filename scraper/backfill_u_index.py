#!/usr/bin/env python3
"""
UmaEdge — U-Index Backfill Utility.

Finds recent races in the database where `u_index` is missing
and uses the Umanity scraper to fetch and update them.

Usage:
    python -m scraper.backfill_u_index --days 30
    python -m scraper.backfill_u_index --date 2026-03-07
    python -m scraper.backfill_u_index --year 2024
"""

import argparse
import logging
from datetime import datetime, timedelta
from sqlalchemy import text

from scraper.db import get_session
from scraper.umanity import scrape_date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_u_index")


def get_dates_missing_u_index(days_back: int = None, year: int = None) -> list[str]:
    """Find distinct dates where entries are missing u_index."""
    with get_session() as session:
        if year:
            query = text("""
                SELECT DISTINCT r.date
                FROM races r
                JOIN entries e ON r.id = e.race_id
                WHERE e.u_index IS NULL
                  AND r.date LIKE :year_prefix
                ORDER BY r.date DESC
            """)
            params = {"year_prefix": f"{year}%"}
        else:
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            query = text("""
                SELECT DISTINCT r.date
                FROM races r
                JOIN entries e ON r.id = e.race_id
                WHERE e.u_index IS NULL
                  AND r.date >= :cutoff_date
                ORDER BY r.date DESC
            """)
            params = {"cutoff_date": cutoff_date}
            
        rows = session.execute(query, params).fetchall()
        
    return [r[0] for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Backfill U-Index from Umanity")
    parser.add_argument("--days", type=int, help="Number of days to look back for missing data")
    parser.add_argument("--date", type=str, help="Specific date to backfill (YYYYMMDD)")
    parser.add_argument("--year", type=int, help="Specific year to backfill (e.g. 2024)")
    args = parser.parse_args()

    if args.date:
        dates_to_process = [args.date]
    elif args.year:
        log.info(f"Scanning for missing U-Index for year {args.year}...")
        dates_to_process = get_dates_missing_u_index(year=args.year)
    else:
        days = args.days or 14
        log.info(f"Scanning for missing U-Index in the last {days} days...")
        dates_to_process = get_dates_missing_u_index(days_back=days)
        
    if not dates_to_process:
        log.info("No dates found with missing U-Index data.")
        return
        
    log.info(f"Found {len(dates_to_process)} dates to backfill: {dates_to_process}")
    
    for d in dates_to_process:
        log.info(f"Backfilling {d} via Umanity...")
        # We use force=True to bypass the 'already in DB' check, allowing save_race_to_db 
        # to execute ON CONFLICT DO UPDATE on the entries table.
        # We skip pedigree to save requests, as we only need the U-Index.
        scrape_date(d, skip_pedigree=True, force=True)

if __name__ == "__main__":
    main()
