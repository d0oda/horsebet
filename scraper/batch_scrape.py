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
import time
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
    Scrape races for a year using hierarchical probing to skip nonexistent
    course/meeting/round combinations efficiently.

    Strategy:
        For each course → meeting → round, probe race #01 first.
        - If race #01 doesn't exist, skip the entire round (12 races).
        - If round 01 of a meeting is empty, skip the entire meeting.
        This avoids wasting ~3s per request on thousands of nonexistent IDs.

    Args:
        year: Year to scrape (e.g. 2024)
        courses: List of JRA course codes (e.g. ["05", "06"])
        dry_run: If True, only show which IDs would be tried
        max_races: Stop after scraping this many new races
        with_odds: Also fetch final odds for each race
        workers: Number of concurrent workers (default: 1, used for races within a valid round)

    Returns:
        Summary dict with counts
    """
    if courses is None:
        courses = JRA_COURSE_CODES

    already_scraped = get_already_scraped_ids()
    log.info(f"Already scraped: {len(already_scraped)} races in DB")

    stats = {
        "total_candidates": 0,
        "already_scraped": 0,
        "races_scraped": 0,
        "races_not_found": 0,
        "races_failed": 0,
        "rounds_skipped": 0,
        "meetings_skipped": 0,
    }

    if dry_run:
        all_ids = generate_race_ids(year, courses)
        new_ids = [rid for rid in all_ids if rid not in already_scraped]
        log.info(f"[DRY RUN] Would try {len(new_ids)} race IDs (with smart skipping, far fewer requests)")
        _print_summary(stats)
        return stats

    def _reached_limit():
        return max_races and stats["races_scraped"] >= max_races

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3
    RATE_LIMIT_PAUSE = 300  # 5 minutes

    for course in courses:
        if _reached_limit():
            break

        log.info(f"📍 Course {course}")

        for meeting in range(1, 7):
            if _reached_limit():
                break

            # Probe: try round 01, race 01 of this meeting
            probe_id = f"{year}{course}{meeting:02d}0101"
            if probe_id in already_scraped:
                # Meeting exists (we scraped it before), check all rounds
                log.info(f"  Meeting {meeting:02d}: already has data, checking rounds...")
                consecutive_failures = 0  # Reset — DB has data for this course
            else:
                # Probe the first race of the first round
                _, probe_status, _ = _scrape_one(probe_id, with_odds)
                if probe_status == "scraped":
                    stats["races_scraped"] += 1
                    consecutive_failures = 0
                    log.info(f"  Meeting {meeting:02d}: ✅ found (probe {probe_id})")
                elif probe_status == "not_found":
                    stats["races_not_found"] += 1
                    consecutive_failures = 0  # not_found is normal, not a failure
                    # Skip this entire meeting
                    skipped = 6 * 12 - 1  # all rounds × races minus the probe
                    stats["meetings_skipped"] += 1
                    log.info(f"  Meeting {meeting:02d}: empty, skipping ({skipped} IDs)")
                    continue
                else:
                    stats["races_failed"] += 1
                    consecutive_failures += 1
                    log.warning(f"  Meeting {meeting:02d}: probe failed ({consecutive_failures} consecutive failures)")

                    # Rate limit detection
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        log.warning(
                            f"⚠️  {consecutive_failures} consecutive probe failures — "
                            f"likely rate-limited. Pausing {RATE_LIMIT_PAUSE}s..."
                        )
                        time.sleep(RATE_LIMIT_PAUSE)
                        # Retry this same probe after pause
                        _, retry_status, _ = _scrape_one(probe_id, with_odds)
                        if retry_status == "scraped":
                            stats["races_scraped"] += 1
                            consecutive_failures = 0
                            log.info(f"  Meeting {meeting:02d}: ✅ recovered after pause")
                        elif retry_status == "not_found":
                            stats["races_not_found"] += 1
                            consecutive_failures = 0
                            stats["meetings_skipped"] += 1
                            log.info(f"  Meeting {meeting:02d}: empty after retry, skipping")
                            continue
                        else:
                            log.error("🛑 Still failing after pause — aborting scrape")
                            _print_summary(stats)
                            return stats
                    continue

            for rnd in range(1, 7):
                if _reached_limit():
                    break

                # Build list of race IDs for this round
                round_ids = []
                for race_num in range(1, 13):
                    rid = f"{year}{course}{meeting:02d}{rnd:02d}{race_num:02d}"
                    if rid == probe_id:
                        continue  # Already probed above
                    if rid in already_scraped:
                        stats["already_scraped"] += 1
                        continue
                    round_ids.append(rid)

                if not round_ids:
                    continue  # All already scraped in this round

                # Probe first race of this round (if not round 01 which was already probed)
                if rnd > 1:
                    first_id = round_ids[0]
                    _, first_status, first_desc = _scrape_one(first_id, with_odds)
                    if first_status == "scraped":
                        stats["races_scraped"] += 1
                        log.info(f"    Round {rnd:02d}: ✅ {first_id} — {first_desc}")
                    elif first_status == "not_found":
                        stats["races_not_found"] += 1
                        stats["rounds_skipped"] += 1
                        log.info(f"    Round {rnd:02d}: empty, skipping {len(round_ids) - 1} remaining")
                        continue
                    else:
                        stats["races_failed"] += 1
                        continue
                    round_ids = round_ids[1:]  # Remove the probed one

                # Scrape remaining races in this round
                if workers > 1 and len(round_ids) > 1:
                    with ThreadPoolExecutor(max_workers=min(workers, len(round_ids))) as pool:
                        futures = {pool.submit(_scrape_one, rid, with_odds): rid for rid in round_ids}
                        for future in as_completed(futures):
                            if _reached_limit():
                                break
                            rid, status, desc = future.result()
                            if status == "scraped":
                                stats["races_scraped"] += 1
                                log.info(f"    [{stats['races_scraped']}/{max_races or '∞'}] ✅ {rid} — {desc}")
                            elif status == "failed":
                                stats["races_failed"] += 1
                                log.error(f"    ❌ {rid}: {desc}")
                            else:
                                stats["races_not_found"] += 1
                else:
                    for rid in round_ids:
                        if _reached_limit():
                            break
                        rid, status, desc = _scrape_one(rid, with_odds)
                        if status == "scraped":
                            stats["races_scraped"] += 1
                            log.info(f"    [{stats['races_scraped']}/{max_races or '∞'}] ✅ {rid} — {desc}")
                        elif status == "failed":
                            stats["races_failed"] += 1
                            log.error(f"    ❌ {rid}: {desc}")
                        else:
                            stats["races_not_found"] += 1

            done = stats["races_scraped"] + stats["races_not_found"] + stats["races_failed"]
            log.info(
                f"  Progress: ✅ {stats['races_scraped']} scraped | "
                f"🔍 {stats['races_not_found']} empty | "
                f"❌ {stats['races_failed']} failed | "
                f"⏭️ {stats['meetings_skipped']} meetings + {stats['rounds_skipped']} rounds skipped"
            )

    _print_summary(stats)
    return stats


def _print_summary(stats: dict):
    """Print a summary of the batch scrape."""
    print("\n" + "=" * 50)
    print("  UmaEdge — Batch Scrape Summary")
    print("=" * 50)
    print(f"  Already scraped:     {stats.get('already_scraped', 0):>6}")
    print(f"  Races scraped:       {stats['races_scraped']:>6}")
    print(f"  Not found (empty):   {stats['races_not_found']:>6}")
    print(f"  Failed:              {stats['races_failed']:>6}")
    print(f"  Meetings skipped:    {stats.get('meetings_skipped', 0):>6}")
    print(f"  Rounds skipped:      {stats.get('rounds_skipped', 0):>6}")
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
