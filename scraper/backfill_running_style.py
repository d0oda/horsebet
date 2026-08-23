"""
UmaEdge — Backfill Running Style from Corner Positions.

Infers running_style (逃/先/差/追) from existing corner_positions data.
No scraping needed — pure DB computation.

Running style classification (JRA conventions):
    逃 (nige / pace-setter):  Led from the front, first corner position = 1-2
    先 (senkou / stalker):    Raced near the front, avg corner ≤ field_size * 0.33
    差 (sashi / closer):      Mid-pack, avg corner ≤ field_size * 0.66
    追 (oikomi / deep closer): Raced from the rear

Usage:
    python -m scraper.backfill_running_style            # backfill all missing
    python -m scraper.backfill_running_style --dry-run  # preview only
"""

import argparse
import logging
import re
from typing import Optional

from sqlalchemy import text

from scraper.db import get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_running_style")


def infer_running_style(corner_positions: str, field_size: int) -> Optional[str]:
    """
    Infer running style from corner position string and field size.

    corner_positions: e.g. '3-3-2-1' or '1-1-1-1'
    field_size: number of runners in the race

    Returns: '逃', '先', '差', '追', or None
    """
    if not corner_positions or not field_size or field_size < 2:
        return None

    # Parse corner positions — format: 'N-N-N-N' (usually 4 corners)
    positions = []
    for p in re.split(r'[-,]', corner_positions.strip()):
        p = p.strip()
        if p.isdigit():
            positions.append(int(p))

    if not positions:
        return None

    first_corner = positions[0]
    avg_pos = sum(positions) / len(positions)

    # Normalize by field size for fair comparison across different field sizes
    # Using fractional position (0.0 = front, 1.0 = back)
    frac = avg_pos / field_size

    # Classification thresholds
    if first_corner <= 2 and frac <= 0.20:
        return '逃'  # Pace-setter: led from the front
    elif frac <= 0.33:
        return '先'  # Stalker: near the front
    elif frac <= 0.66:
        return '差'  # Closer: mid-pack
    else:
        return '追'  # Deep closer: from the rear


def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Backfill Running Style")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated")
    args = parser.parse_args()

    # Get results with corner_positions but no running_style
    with get_session() as session:
        results = session.execute(text("""
            SELECT r.id, r.corner_positions, r.finish_pos,
                   ra.field_size
            FROM results r
            JOIN entries e ON e.id = r.entry_id
            JOIN races ra ON ra.id = e.race_id
            WHERE r.running_style IS NULL
              AND r.corner_positions IS NOT NULL
              AND r.corner_positions != ''
            ORDER BY r.id
        """)).fetchall()

    total = len(results)
    log.info(f"Found {total} results with corner positions but no running_style")

    if not results:
        log.info("Nothing to backfill!")
        return

    # Classify in batches
    updates = []
    skipped = 0
    style_counts = {'逃': 0, '先': 0, '差': 0, '追': 0}

    for result_id, corners, finish_pos, field_size in results:
        style = infer_running_style(corners, field_size)
        if style:
            updates.append((result_id, style))
            style_counts[style] += 1
        else:
            skipped += 1

    log.info(f"Classified {len(updates)} results, {skipped} could not be classified")
    log.info(f"  Distribution: 逃={style_counts['逃']}, 先={style_counts['先']}, "
             f"差={style_counts['差']}, 追={style_counts['追']}")

    if args.dry_run:
        log.info("(dry run — no DB changes made)")
        return

    # Batch update in chunks of 500
    BATCH_SIZE = 500
    updated = 0
    with get_session() as session:
        for i in range(0, len(updates), BATCH_SIZE):
            batch = updates[i:i + BATCH_SIZE]
            for result_id, style in batch:
                session.execute(
                    text("UPDATE results SET running_style = :style WHERE id = :rid"),
                    {"style": style, "rid": result_id},
                )
            updated += len(batch)
            if updated % 5000 == 0 or updated == len(updates):
                log.info(f"  Updated {updated}/{len(updates)}...")

    log.info(f"\n{'=' * 50}")
    log.info(f"Backfill complete: {updated} running styles updated, {skipped} skipped")
    log.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
