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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from sqlalchemy import text

from scraper.db import get_session
from scraper.netkeiba import scrape_race
from scraper.odds_watcher import fetch_win_odds, save_odds_snapshot

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

def _scrape_one(rid: str, with_odds: bool) -> tuple[str, str, str]:
    """Scrape a single race ID. Returns (rid, status, description)."""
    try:
        result = scrape_race(rid)
        if result and result.entries:
            desc = f"{result.race_name_jp} ({len(result.entries)} entries)"
            # Optionally scrape final odds snapshot
            if with_odds:
                odds = fetch_win_odds(rid)
                if odds:
                    count = save_odds_snapshot(rid, odds)
                    desc += f" + {count} odds"
            return (rid, "scraped", desc)
        else:
            return (rid, "not_found", "")
    except Exception as e:
        return (rid, "failed", str(e))


def batch_scrape(
    year: int,
    courses: Optional[list[str]] = None,
    dry_run: bool = False,
    max_races: Optional[int] = None,
    with_odds: bool = False,
    workers: int = 1,
) -> dict:
    """
    Scrape races for a year by trying generated JRA race IDs.

    Args:
        year: Year to scrape (e.g. 2024)
        courses: List of JRA course codes (e.g. ["05", "06"])
        dry_run: If True, only show which IDs would be tried
        max_races: Stop after scraping this many new races
        with_odds: Also fetch final odds for each race
        workers: Number of concurrent workers (default: 1)

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

    if workers <= 1:
        # Sequential mode (original behavior)
        for i, rid in enumerate(new_ids):
            if max_races and stats["races_scraped"] >= max_races:
                log.info(f"Reached max_races limit ({max_races}). Stopping.")
                break

            rid, status, desc = _scrape_one(rid, with_odds)
            if status == "scraped":
                stats["races_scraped"] += 1
                log.info(f"  [{stats['races_scraped']}/{max_races or '∞'}] ✅ {rid} — {desc}")
            elif status == "failed":
                stats["races_failed"] += 1
                log.error(f"  ❌ Failed {rid}: {desc}")
            else:
                stats["races_not_found"] += 1
    else:
        # Concurrent mode
        log.info(f"Using {workers} concurrent workers")
        lock = threading.Lock()
        stop_event = threading.Event()

        def _worker(rid: str) -> tuple[str, str, str]:
            if stop_event.is_set():
                return (rid, "skipped", "")
            return _scrape_one(rid, with_odds)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_worker, rid): rid
                for rid in new_ids
            }
            for future in as_completed(futures):
                rid, status, desc = future.result()
                with lock:
                    if status == "scraped":
                        stats["races_scraped"] += 1
                        log.info(
                            f"  [{stats['races_scraped']}/{max_races or '∞'}] ✅ {rid} — {desc}"
                        )
                        if max_races and stats["races_scraped"] >= max_races:
                            log.info(f"Reached max_races limit ({max_races}). Stopping.")
                            stop_event.set()
                    elif status == "failed":
                        stats["races_failed"] += 1
                        log.error(f"  ❌ Failed {rid}: {desc}")
                    elif status == "not_found":
                        stats["races_not_found"] += 1

                    done = stats["races_scraped"] + stats["races_not_found"] + stats["races_failed"]
                    if done % 100 == 0:
                        log.info(
                            f"  Progress: {done}/{len(new_ids)} "
                            f"(✅ {stats['races_scraped']} | ❌ {stats['races_failed']} | 🔍 {stats['races_not_found']} 404s)"
                        )

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
    parser.add_argument("--year", type=int, help="Year to scrape (e.g. 2024)")
    parser.add_argument("--years", type=str, help="Comma-separated years (e.g. 2022,2023,2024)")
    parser.add_argument("--courses", type=str, help="Comma-separated course codes (e.g. 05,06)")
    parser.add_argument("--dry-run", action="store_true", help="Only show candidate IDs, don't scrape")
    parser.add_argument("--max-races", type=int, default=50, help="Stop after N new races per year (default: 50)")
    parser.add_argument("--with-odds", action="store_true", help="Also scrape final odds for each race")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers (default: 1)")

    args = parser.parse_args()
    courses = args.courses.split(",") if args.courses else None

    if args.years:
        years = [int(y.strip()) for y in args.years.split(",")]
    elif args.year:
        years = [args.year]
    else:
        parser.error("Either --year or --years is required")
        return

    total_stats = {"races_scraped": 0, "races_not_found": 0, "races_failed": 0}
    for year in years:
        log.info(f"\n{'=' * 50}")
        log.info(f"  Scraping year {year}")
        log.info(f"{'=' * 50}")
        stats = batch_scrape(year, courses=courses, dry_run=args.dry_run, max_races=args.max_races, with_odds=args.with_odds, workers=args.workers)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    if len(years) > 1:
        print(f"\n{'=' * 50}")
        print(f"  Multi-Year Total ({', '.join(str(y) for y in years)})")
        print(f"{'=' * 50}")
        print(f"  Races scraped:   {total_stats['races_scraped']:>6}")
        print(f"  Not found (404): {total_stats['races_not_found']:>6}")
        print(f"  Failed:          {total_stats['races_failed']:>6}")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
