"""
UmaEdge — Batch Historical Race Scraper.

Systematically scrapes JRA race results from netkeiba.com by generating
race IDs from the known JRA numbering scheme: YYYYCCDDRRNN
  YYYY = year, CC = course (01-10), DD = meeting (01-06),
  RR = round (01-06), NN = race number (01-12)

Usage:
    # Scrape all 2024 races (capped at 50 new races)
    python -m scraper.batch_scrape --year 2024 --max-races 50

    # Scrape only Tokyo (05) and Nakayama (06)
    python -m scraper.batch_scrape --year 2024 --courses 05,06 --max-races 30

    # Dry run — check what IDs would be generated
    python -m scraper.batch_scrape --year 2024 --dry-run

    # Resume (skips already-scraped races automatically)
    python -m scraper.batch_scrape --year 2024
"""

import argparse
import logging
from typing import Optional

from sqlalchemy import text

from scraper.db import get_session
from scraper.netkeiba import scrape_race

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_scrape")

# JRA courses
JRA_COURSE_CODES = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]


# ---------------------------------------------------------------------------
# Race ID Generation
# ---------------------------------------------------------------------------

def generate_race_ids(
    year: int,
    courses: Optional[list[str]] = None,
    meetings: range = range(1, 7),    # 01-06 meeting days per course per year
    rounds: range = range(1, 7),      # 01-06 rounds per meeting
    race_nums: range = range(1, 13),  # 01-12 races per round day
) -> list[str]:
    """
    Generate candidate JRA race IDs for a given year.
    Format: YYYYCCDDRRNN
    """
    if courses is None:
        courses = JRA_COURSE_CODES

    ids = []
    for course in courses:
        for meeting in meetings:
            for rnd in rounds:
                for race in race_nums:
                    race_id = f"{year}{course}{meeting:02d}{rnd:02d}{race:02d}"
                    ids.append(race_id)
    return ids


def get_already_scraped_ids() -> set[str]:
    """Return the set of netkeiba race IDs already in the database."""
    with get_session() as session:
        result = session.execute(text("SELECT netkeiba_id FROM races"))
        return {row[0] for row in result.fetchall()}


# ---------------------------------------------------------------------------
# Batch Scraping
# ---------------------------------------------------------------------------

def batch_scrape(
    year: int,
    courses: Optional[list[str]] = None,
    dry_run: bool = False,
    max_races: Optional[int] = None,
) -> dict:
    """
    Scrape races for a year by trying generated JRA race IDs.

    Args:
        year: Year to scrape (e.g. 2024)
        courses: List of JRA course codes (e.g. ["05", "06"])
        dry_run: If True, only show which IDs would be tried
        max_races: Stop after scraping this many new races

    Returns:
        Summary dict with counts
    """
    race_ids = generate_race_ids(year, courses)
    log.info(f"Generated {len(race_ids)} candidate race IDs for {year}")

    already_scraped = get_already_scraped_ids()
    log.info(f"Already scraped: {len(already_scraped)} races in DB")

    # Filter out already-scraped
    new_ids = [rid for rid in race_ids if rid not in already_scraped]
    log.info(f"New IDs to try: {len(new_ids)} (skipping {len(race_ids) - len(new_ids)} existing)")

    stats = {
        "total_candidates": len(race_ids),
        "already_scraped": len(race_ids) - len(new_ids),
        "races_scraped": 0,
        "races_not_found": 0,
        "races_failed": 0,
    }

    if dry_run:
        log.info(f"[DRY RUN] Would try {len(new_ids)} race IDs")
        for rid in new_ids[:20]:
            log.info(f"  🆕 {rid}")
        if len(new_ids) > 20:
            log.info(f"  ... and {len(new_ids) - 20} more")
        _print_summary(stats)
        return stats

    for i, rid in enumerate(new_ids):
        if max_races and stats["races_scraped"] >= max_races:
            log.info(f"Reached max_races limit ({max_races}). Stopping.")
            break

        try:
            result = scrape_race(rid)
            if result and result.entries:
                stats["races_scraped"] += 1
                log.info(
                    f"  [{stats['races_scraped']}/{max_races or '∞'}] ✅ {rid}"
                    f" — {result.race_name_jp} ({len(result.entries)} entries)"
                )
            else:
                stats["races_not_found"] += 1
        except Exception as e:
            log.error(f"  ❌ Failed {rid}: {e}")
            stats["races_failed"] += 1

    _print_summary(stats)
    return stats


def _print_summary(stats: dict):
    """Print a summary of the batch scrape."""
    print("\n" + "=" * 50)
    print("  UmaEdge — Batch Scrape Summary")
    print("=" * 50)
    print(f"  Total candidates:    {stats['total_candidates']:>6}")
    print(f"  Already scraped:     {stats['already_scraped']:>6}")
    print(f"  Races scraped:       {stats['races_scraped']:>6}")
    print(f"  Not found (404):     {stats['races_not_found']:>6}")
    print(f"  Failed:              {stats['races_failed']:>6}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Batch Historical Scraper")
    parser.add_argument("--year", type=int, required=True, help="Year to scrape (e.g. 2024)")
    parser.add_argument("--courses", type=str, help="Comma-separated course codes (e.g. 05,06)")
    parser.add_argument("--dry-run", action="store_true", help="Only show candidate IDs, don't scrape")
    parser.add_argument("--max-races", type=int, default=50, help="Stop after N new races (default: 50)")

    args = parser.parse_args()
    courses = args.courses.split(",") if args.courses else None
    batch_scrape(args.year, courses=courses, dry_run=args.dry_run, max_races=args.max_races)


if __name__ == "__main__":
    main()
