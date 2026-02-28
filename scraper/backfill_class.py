"""
UmaEdge — Backfill Race Class from Race Names.

Parses existing race names/grades in the DB to fill in missing `class`
for 257 races. No scraping needed — pure SQL/DB work.

JRA race class markers in race names:
  - "新馬" → 新馬 (maiden)
  - "未勝利" → 未勝利 (not-yet-won)
  - "1勝クラス" → 1勝クラス
  - "2勝クラス" → 2勝クラス
  - "3勝クラス" → 3勝クラス
  - "オープン" → オープン (open)
  - Grade markers: "G1", "G2", "G3", "(G)" in name or grade field

Usage:
    python -m scraper.backfill_class            # backfill all missing
    python -m scraper.backfill_class --dry-run  # show what would be updated
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
log = logging.getLogger("backfill_class")

# Patterns to extract class from race name, ordered by specificity
CLASS_PATTERNS = [
    (r"新馬", "新馬"),
    (r"未勝利", "未勝利"),
    (r"1勝クラス", "1勝クラス"),
    (r"2勝クラス", "2勝クラス"),
    (r"3勝クラス", "3勝クラス"),
    # Older naming conventions (pre-2019)
    (r"500万下", "1勝クラス"),
    (r"1000万下", "2勝クラス"),
    (r"1600万下", "3勝クラス"),
    (r"オープン", "オープン"),
]


def infer_class(name: Optional[str], grade: Optional[str]) -> Optional[str]:
    """Infer race class from name and/or grade field."""
    # Grade takes priority
    if grade:
        grade_str = str(grade).strip()
        if grade_str in ("G1", "G2", "G3", "GI", "GII", "GIII"):
            return "オープン"
        if grade_str in ("(G)",):
            return "オープン"
        if grade_str in ("OP", "L", "Listed"):
            return "オープン"

    # Try to match class from race name
    if name:
        for pattern, class_val in CLASS_PATTERNS:
            if re.search(pattern, name):
                return class_val

    return None


def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Backfill Race Class")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated")
    args = parser.parse_args()

    # Get races missing class
    with get_session() as session:
        races = session.execute(text("""
            SELECT id, netkeiba_id, race_name_jp, grade
            FROM races
            WHERE class IS NULL
            ORDER BY id
        """)).fetchall()

    total = len(races)
    log.info(f"Found {total} races with missing class")

    if not races:
        log.info("Nothing to backfill!")
        return

    updated = 0
    failed = 0

    for i, (race_id, nk_id, name, grade) in enumerate(races, 1):
        inferred = infer_class(name, grade)

        if inferred:
            if args.dry_run:
                log.info(f"  [{i}/{total}] race {nk_id}: '{name}' → {inferred}")
            else:
                with get_session() as session:
                    session.execute(
                        text("UPDATE races SET class = :cls WHERE id = :rid"),
                        {"cls": inferred, "rid": race_id},
                    )
            updated += 1
        else:
            if i <= 20 or total <= 50:  # Don't spam for large sets
                log.debug(f"  [{i}/{total}] race {nk_id}: '{name}' (grade={grade}) — could not infer class")
            failed += 1

    log.info(f"\n{'=' * 50}")
    log.info(f"Backfill complete: {updated} updated, {failed} could not infer")
    if args.dry_run:
        log.info("(dry run — no DB changes made)")
    log.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
