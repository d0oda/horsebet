"""
UmaEdge — JRA-VAN DataLab Integration (Optional).

Wrapper for JRA-VAN's JV-Link COM API, which provides official race data
including sectional times, training (調教) data, and real-time odds.

Requirements:
    - JRA-VAN DataLab subscription (~¥2,200/month)
    - Windows environment (JV-Link is a COM/ActiveX component)
    - Alternatively: run on a Windows VM and export CSVs

This module provides:
    1. CSV-based import: read JRA-VAN exported CSV files into the DB
    2. COM stub: placeholder for direct JV-Link integration

Usage:
    # Import from CSV exports (most portable approach)
    python -m scraper.jra_van --csv-dir /path/to/jravan_exports/

    # Direct JV-Link (Windows only, requires pywin32)
    python -m scraper.jra_van --live
"""

import argparse
import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import text

from scraper.db import get_session

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("jra_van")


# ---------------------------------------------------------------------------
# CSV Import — Platform-Independent
# ---------------------------------------------------------------------------

# JRA-VAN CSV column mappings (based on standard DataLab exports)
RACE_CSV_COLS = {
    "RaceID": "netkeiba_id",       # maps to our race ID scheme
    "RaceDate": "date",
    "CourseName": "course_name",
    "RaceNumber": "race_number",
    "Distance": "distance",
    "Surface": "surface",           # 芝=turf, ダート=dirt
    "Going": "going",
    "ClassName": "class",
    "GradeCode": "grade",
    "RaceName": "race_name_jp",
    "Weather": "weather",
    "FieldSize": "field_size",
}

ENTRY_CSV_COLS = {
    "HorseID": "horse_id",
    "HorseName": "horse_name",
    "JockeyName": "jockey_name",
    "Draw": "draw",
    "PostPosition": "post_position",
    "WeightCarried": "weight_carried",
    "HorseWeight": "horse_weight",
    "HorseWeightChange": "horse_weight_change",
    "OddsWin": "odds_win",
    "Popularity": "popularity",
    "FinishPosition": "finish_pos",
    "Margin": "margin",
    "TimeSecs": "time_secs",
    "Last3F": "last_3f_secs",
    "First3F": "first_3f_secs",
    "CornerPositions": "corner_positions",
}

SURFACE_MAP = {"芝": "turf", "ダート": "dirt", "ダ": "dirt"}


def import_race_csv(csv_path: str) -> int:
    """
    Import a JRA-VAN race CSV into the database.

    Expected format: one row per entry (horse-in-race), with race-level
    columns duplicated across rows for the same race.

    Returns count of new entries inserted.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        log.error(f"CSV file not found: {csv_path}")
        return 0

    count = 0
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        current_race_id = None

        with get_session() as session:
            for row in reader:
                race_id_raw = row.get("RaceID", "").strip()
                if not race_id_raw:
                    continue

                # Insert race (once per race)
                if race_id_raw != current_race_id:
                    current_race_id = race_id_raw

                    # Parse date
                    date_str = row.get("RaceDate", "")
                    try:
                        race_date = datetime.strptime(date_str, "%Y%m%d").date()
                    except ValueError:
                        race_date = None

                    # Resolve course
                    course_name = row.get("CourseName", "").strip()
                    course_row = session.execute(
                        text("SELECT id FROM courses WHERE name_jp = :name"),
                        {"name": course_name},
                    ).fetchone()
                    course_id = course_row[0] if course_row else None

                    surface = SURFACE_MAP.get(
                        row.get("Surface", "").strip(), row.get("Surface", "").strip()
                    )

                    # Upsert race
                    session.execute(
                        text("""
                            INSERT INTO races (netkeiba_id, date, course_id, race_number,
                                               distance, surface, going, class, grade,
                                               race_name_jp, weather, field_size)
                            VALUES (:nid, :date, :cid, :rn, :dist, :surf, :going,
                                    :cls, :grade, :name, :weather, :fs)
                            ON CONFLICT (netkeiba_id) DO NOTHING
                        """),
                        {
                            "nid": race_id_raw,
                            "date": race_date,
                            "cid": course_id,
                            "rn": _safe_int(row.get("RaceNumber")),
                            "dist": _safe_int(row.get("Distance")),
                            "surf": surface,
                            "going": row.get("Going", "").strip() or None,
                            "cls": row.get("ClassName", "").strip() or None,
                            "grade": row.get("GradeCode", "").strip() or None,
                            "name": row.get("RaceName", "").strip() or None,
                            "weather": row.get("Weather", "").strip() or None,
                            "fs": _safe_int(row.get("FieldSize")),
                        },
                    )

                # Insert horse (simplified — JRA-VAN gives us an ID)
                horse_name = row.get("HorseName", "").strip()
                horse_jravan_id = row.get("HorseID", "").strip()

                if horse_name and horse_jravan_id:
                    session.execute(
                        text("""
                            INSERT INTO horses (name, name_jp, netkeiba_id)
                            VALUES (:name, :name, :nid)
                            ON CONFLICT (netkeiba_id) DO NOTHING
                        """),
                        {"name": horse_name, "nid": horse_jravan_id},
                    )

                count += 1

    log.info(f"📥 Imported {count} entries from {csv_path.name}")
    return count


def import_csv_directory(csv_dir: str) -> int:
    """Import all CSV files from a directory. Returns total entries imported."""
    csv_dir = Path(csv_dir)
    if not csv_dir.is_dir():
        log.error(f"Directory not found: {csv_dir}")
        return 0

    total = 0
    csv_files = sorted(csv_dir.glob("*.csv"))
    log.info(f"Found {len(csv_files)} CSV files in {csv_dir}")

    for csv_file in csv_files:
        total += import_race_csv(str(csv_file))

    log.info(f"✅ Total entries imported: {total}")
    return total


# ---------------------------------------------------------------------------
# JV-Link COM Interface (Windows Only)
# ---------------------------------------------------------------------------

def jvlink_available() -> bool:
    """Check if JV-Link COM component is available (Windows only)."""
    try:
        import win32com.client  # noqa: F401
        return True
    except ImportError:
        return False


def fetch_via_jvlink(data_type: str = "RACE", from_date: Optional[str] = None):
    """
    Fetch data directly from JV-Link COM API.

    This is a stub — full implementation requires:
    1. Windows environment with pywin32
    2. JRA-VAN DataLab software installed
    3. Active subscription

    Data types:
        RACE  — Race results and entries
        DIFF  — Incremental updates since last fetch
        ODDS  — Real-time odds
        TRAIN — Training (調教) data
    """
    if not jvlink_available():
        log.warning(
            "⚠️  JV-Link not available. This requires:\n"
            "   1. Windows OS\n"
            "   2. pywin32 package (pip install pywin32)\n"
            "   3. JRA-VAN DataLab software installed\n"
            "\n"
            "   Alternative: Export CSVs from DataLab and use --csv-dir import."
        )
        return None

    # Stub for COM interface
    # import win32com.client
    # jvlink = win32com.client.Dispatch("JVDTLab.JVLink.1")
    # jvlink.JVInit("UMAEDGE")
    # ret = jvlink.JVOpen(data_type, from_date or "20230101", 1, 0, 0, 0)
    # ...

    log.info(f"JV-Link fetch for {data_type} — not yet implemented")
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(val) -> Optional[int]:
    """Safely convert a value to int, returning None on failure."""
    if val is None:
        return None
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — JRA-VAN DataLab Integration")
    parser.add_argument(
        "--csv-dir", type=str,
        help="Directory containing JRA-VAN exported CSV files",
    )
    parser.add_argument(
        "--csv", type=str,
        help="Single CSV file to import",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Fetch data directly from JV-Link (Windows only)",
    )

    args = parser.parse_args()

    if args.csv_dir:
        import_csv_directory(args.csv_dir)
    elif args.csv:
        import_race_csv(args.csv)
    elif args.live:
        fetch_via_jvlink()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
