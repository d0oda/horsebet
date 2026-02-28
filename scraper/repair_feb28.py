"""
Repair Feb 28, 2026 race data.

Deletes all garbled Feb 28 entries/results/races, then re-scrapes
them using the fixed parser that correctly handles:
  - race.netkeiba.com HTML structure (shutuba/result pages)
  - EUC-JP encoding
  - draw (枠番) vs post_position (馬番) distinction
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
from sqlalchemy import text

from scraper.db import get_session
from scraper.netkeiba import scrape_race

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("repair_feb28")

TARGET_DATE = "2026-02-28"


def delete_feb28_data():
    """Delete all entries, results, and races for the target date."""
    with get_session() as session:
        # Get race IDs
        races = session.execute(
            text("SELECT id, netkeiba_id FROM races WHERE date = :d"),
            {"d": TARGET_DATE},
        ).fetchall()

        if not races:
            log.info(f"No races found for {TARGET_DATE}")
            return []

        race_ids = [r[0] for r in races]
        nk_ids = [r[1] for r in races]
        log.info(f"Found {len(races)} races to delete for {TARGET_DATE}")

        # Delete results (via entries)
        for rid in race_ids:
            session.execute(
                text("DELETE FROM results WHERE entry_id IN (SELECT id FROM entries WHERE race_id = :rid)"),
                {"rid": rid},
            )

        # Delete entries
        for rid in race_ids:
            session.execute(
                text("DELETE FROM entries WHERE race_id = :rid"),
                {"rid": rid},
            )

        # Delete races
        session.execute(
            text("DELETE FROM races WHERE date = :d"),
            {"d": TARGET_DATE},
        )

        session.commit()
        log.info(f"✅ Deleted {len(races)} races and all associated entries/results")

        return nk_ids


def rescrape_races(nk_ids: list[str]):
    """Re-scrape all races using the fixed parser."""
    log.info(f"Re-scraping {len(nk_ids)} races...")

    success = 0
    failed = 0
    for i, nk_id in enumerate(nk_ids, 1):
        log.info(f"  [{i}/{len(nk_ids)}] {nk_id}...")
        try:
            result = scrape_race(nk_id)
            if result and result.entries:
                success += 1
                log.info(
                    f"    ✅ {result.race_name_jp} — {len(result.entries)} entries, "
                    f"dist={result.distance}m, surface={result.surface}"
                )
            else:
                failed += 1
                log.warning(f"    ❌ No data returned")
        except Exception as e:
            failed += 1
            log.error(f"    ❌ Error: {e}")

        # Brief pause between races
        time.sleep(0.5)

    log.info(f"\n{'='*50}")
    log.info(f"  Repair Summary")
    log.info(f"{'='*50}")
    log.info(f"  Success: {success}")
    log.info(f"  Failed:  {failed}")
    log.info(f"{'='*50}")


def verify_data():
    """Verify the repaired data looks correct."""
    with get_session() as session:
        races = session.execute(
            text("""
                SELECT r.id, r.netkeiba_id, r.race_number, r.race_name_jp, 
                       r.distance, r.surface, c.name as venue
                FROM races r LEFT JOIN courses c ON r.course_id = c.id
                WHERE r.date = :d ORDER BY c.name, r.race_number
            """),
            {"d": TARGET_DATE},
        ).fetchall()

        log.info(f"\n{'='*60}")
        log.info(f"  Verification: {len(races)} races for {TARGET_DATE}")
        log.info(f"{'='*60}")

        issues = 0
        for r in races:
            # Check for garbled names
            name = r[3] or ""
            is_garbled = any(ord(c) > 65000 for c in name) or "?" in name

            # Check entries
            entries = session.execute(
                text("""
                    SELECT e.draw, e.post_position, h.name_jp, e.odds_win, e.popularity
                    FROM entries e JOIN horses h ON e.horse_id = h.id
                    WHERE e.race_id = :rid ORDER BY e.post_position
                """),
                {"rid": r[0]},
            ).fetchall()

            # Check for duplicate post_positions
            post_positions = [e[1] for e in entries]
            has_dupes = len(post_positions) != len(set(post_positions))

            # Check for garbled horse names
            garbled_horses = sum(1 for e in entries if any(ord(c) > 65000 for c in (e[2] or "")))

            status = "✅" if not is_garbled and not has_dupes and garbled_horses == 0 and r[4] and r[4] > 0 else "❌"
            if status == "❌":
                issues += 1

            log.info(
                f"  {status} {r[6]} R{r[2]}: {name[:20]:<20s} "
                f"dist={r[4]}m {r[5]} entries={len(entries)} "
                f"{'GARBLED_NAME' if is_garbled else ''}"
                f"{'DUPE_PP' if has_dupes else ''}"
                f"{f'GARBLED_HORSES({garbled_horses})' if garbled_horses else ''}"
            )

        if issues == 0:
            log.info(f"\n  ✅ All {len(races)} races look clean!")
        else:
            log.info(f"\n  ⚠️  {issues} races have issues")


if __name__ == "__main__":
    log.info(f"🔧 Repairing {TARGET_DATE} race data...")

    # Step 1: Delete garbled data
    nk_ids = delete_feb28_data()

    if nk_ids:
        # Step 2: Re-scrape with fixed parser
        rescrape_races(nk_ids)

        # Step 3: Verify
        verify_data()
    else:
        log.info("Nothing to repair.")
