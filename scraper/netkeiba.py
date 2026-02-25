"""
UmaEdge — Netkeiba Race Scraper.

Scrapes race results and horse data from netkeiba.com.
Stores parsed data into the horsebet schema on Supabase.

Usage:
    # Scrape a single race by netkeiba race ID
    python -m scraper.netkeiba --race 202506010101

    # Scrape all G1-G3 races for a year
    python -m scraper.netkeiba --year 2025 --grades G1,G2,G3

    # Scrape race list for a specific date
    python -m scraper.netkeiba --date 2025-12-28
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy import text

from scraper.db import get_session

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("netkeiba")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://db.netkeiba.com"
RACE_URL = BASE_URL + "/race/{race_id}/"
HORSE_URL = BASE_URL + "/horse/{horse_id}/"
RACE_LIST_URL = "https://race.netkeiba.com/top/race_list.html?kaisai_date={date}"

DELAY_MIN = float(os.getenv("SCRAPE_DELAY_MIN", 2))
DELAY_MAX = float(os.getenv("SCRAPE_DELAY_MAX", 5))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

# JRA course mapping
JRA_COURSES = {
    "01": {"name": "Sapporo", "name_jp": "札幌", "surface": "turf", "direction": "right"},
    "02": {"name": "Hakodate", "name_jp": "函館", "surface": "turf", "direction": "right"},
    "03": {"name": "Fukushima", "name_jp": "福島", "surface": "turf", "direction": "right"},
    "04": {"name": "Niigata", "name_jp": "新潟", "surface": "turf", "direction": "left"},
    "05": {"name": "Tokyo", "name_jp": "東京", "surface": "turf", "direction": "left"},
    "06": {"name": "Nakayama", "name_jp": "中山", "surface": "turf", "direction": "right"},
    "07": {"name": "Chukyo", "name_jp": "中京", "surface": "turf", "direction": "left"},
    "08": {"name": "Kyoto", "name_jp": "京都", "surface": "turf", "direction": "right"},
    "09": {"name": "Hanshin", "name_jp": "阪神", "surface": "turf", "direction": "right"},
    "10": {"name": "Kokura", "name_jp": "小倉", "surface": "turf", "direction": "right"},
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class HorseData:
    name: str
    name_jp: str
    sex: str
    birth_year: Optional[int]
    netkeiba_id: str
    owner: Optional[str] = None
    sire_name: Optional[str] = None
    dam_name: Optional[str] = None


@dataclass
class EntryData:
    post_position: int                  # 馬番
    draw: Optional[int]                 # 枠番
    horse: HorseData
    jockey_name: Optional[str] = None
    jockey_name_jp: Optional[str] = None
    weight_carried: Optional[float] = None
    horse_weight: Optional[int] = None
    horse_weight_change: Optional[int] = None
    odds_win: Optional[float] = None
    popularity: Optional[int] = None
    # Result fields (filled after race)
    finish_pos: Optional[int] = None
    margin: Optional[str] = None
    time_secs: Optional[float] = None
    last_3f_secs: Optional[float] = None
    corner_positions: Optional[str] = None


@dataclass
class RaceData:
    netkeiba_id: str
    race_name: Optional[str] = None
    race_name_jp: Optional[str] = None
    date: Optional[str] = None          # YYYY-MM-DD
    course_code: Optional[str] = None   # JRA course code (01-10)
    race_number: Optional[int] = None
    distance: Optional[int] = None
    surface: Optional[str] = None       # 'turf' or 'dirt'
    going: Optional[str] = None         # 良/稍重/重/不良
    class_: Optional[str] = None
    grade: Optional[str] = None
    weather: Optional[str] = None
    field_size: Optional[int] = None
    entries: list[EntryData] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP Helpers
# ---------------------------------------------------------------------------

def _sleep():
    """Random delay between requests to be respectful."""
    delay = random.uniform(DELAY_MIN, DELAY_MAX)
    time.sleep(delay)


def _fetch(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch a URL and return parsed BeautifulSoup, with retries."""
    for attempt in range(retries):
        try:
            _sleep()
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.encoding = "EUC-JP"  # netkeiba uses EUC-JP
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            elif resp.status_code == 404:
                log.warning(f"404 Not Found: {url}")
                return None
            else:
                log.warning(f"HTTP {resp.status_code} for {url} (attempt {attempt + 1})")
        except requests.RequestException as e:
            log.warning(f"Request error for {url}: {e} (attempt {attempt + 1})")

        if attempt < retries - 1:
            time.sleep(5 * (attempt + 1))

    log.error(f"Failed to fetch {url} after {retries} attempts")
    return None


# ---------------------------------------------------------------------------
# Parsing — Race Result Page
# ---------------------------------------------------------------------------

def _parse_time(time_str: str) -> Optional[float]:
    """Convert race time string like '1:34.5' to seconds."""
    if not time_str or time_str.strip() == "":
        return None
    try:
        time_str = time_str.strip()
        if ":" in time_str:
            parts = time_str.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(time_str)
    except (ValueError, IndexError):
        return None


def _parse_weight(weight_str: str) -> tuple[Optional[int], Optional[int]]:
    """Parse horse weight string like '480(+4)' into (weight, change)."""
    if not weight_str:
        return None, None
    match = re.match(r"(\d+)\(([+-]?\d+)\)", weight_str.strip())
    if match:
        return int(match.group(1)), int(match.group(2))
    try:
        return int(weight_str.strip()), None
    except ValueError:
        return None, None


def _extract_horse_id(href: str) -> Optional[str]:
    """Extract horse ID from a netkeiba URL like '/horse/2019104308/'."""
    match = re.search(r"/horse/(\w+)", href)
    return match.group(1) if match else None


def _extract_jockey_name(cell) -> tuple[Optional[str], Optional[str]]:
    """Extract jockey name (romanised) and Japanese from a table cell."""
    link = cell.find("a")
    if link:
        name_jp = link.get_text(strip=True)
        # Romanised name not always available on result page
        return None, name_jp
    return None, cell.get_text(strip=True) if cell else (None, None)


def parse_race_page(soup: BeautifulSoup, race_id: str) -> Optional[RaceData]:
    """Parse a netkeiba race result page into a RaceData object."""
    race = RaceData(netkeiba_id=race_id)

    # --- Race header info ---
    header = soup.find("div", class_="data_intro")
    if not header:
        log.warning(f"No race header found for {race_id}")
        return None

    # Race name
    title = header.find("h1")
    if title:
        race.race_name_jp = title.get_text(strip=True)
        race.race_name = race.race_name_jp  # Could transliterate later

    # Race details line (e.g., "芝右 2000m / 天候 : 晴 / 芝 : 良")
    detail_span = header.find("span")
    if detail_span:
        detail_text = detail_span.get_text(strip=True)

        # Distance
        dist_match = re.search(r"(\d{3,4})m", detail_text)
        if dist_match:
            race.distance = int(dist_match.group(1))

        # Surface
        if "ダート" in detail_text or "ダ" in detail_text:
            race.surface = "dirt"
        elif "芝" in detail_text:
            race.surface = "turf"

        # Going / track condition
        going_match = re.search(r"[芝ダート]+\s*:\s*(良|稍重|重|不良)", detail_text)
        if going_match:
            race.going = going_match.group(1)

        # Weather
        weather_match = re.search(r"天候\s*:\s*(\S+)", detail_text)
        if weather_match:
            race.weather = weather_match.group(1)

    # Date and course from race ID
    # Race ID format: YYYYCCDDRRNN (Year, Course, Day, Round, RaceNo)
    if len(race_id) >= 12:
        race.date = f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}" if race_id[4:8].isdigit() else None
        race.course_code = race_id[4:6]
        try:
            race.race_number = int(race_id[10:12])
        except (ValueError, IndexError):
            pass

    # Grade detection
    name_text = race.race_name_jp or ""
    for g in ["G1", "G2", "G3", "GI", "GII", "GIII", "Ｇ１", "Ｇ２", "Ｇ３"]:
        if g in name_text:
            race.grade = g.replace("Ｇ", "G").replace("I", "1").replace("II", "2").replace("III", "3")
            break

    # --- Parse result table ---
    result_table = soup.find("table", class_="race_table_01")
    if not result_table:
        log.warning(f"No result table found for {race_id}")
        return race

    rows = result_table.find_all("tr")[1:]  # skip header
    race.field_size = len(rows)

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 13:
            continue

        try:
            # Finish position
            finish_text = cells[0].get_text(strip=True)
            finish_pos = int(finish_text) if finish_text.isdigit() else None

            # Draw (枠番) and post position (馬番)
            draw = int(cells[1].get_text(strip=True)) if cells[1].get_text(strip=True).isdigit() else None
            post_pos = int(cells[2].get_text(strip=True)) if cells[2].get_text(strip=True).isdigit() else None

            # Horse
            horse_cell = cells[3]
            horse_link = horse_cell.find("a")
            horse_name_jp = horse_link.get_text(strip=True) if horse_link else cells[3].get_text(strip=True)
            horse_id = _extract_horse_id(horse_link["href"]) if horse_link and horse_link.get("href") else None

            # Sex and age (e.g. "牡3")
            sex_age = cells[4].get_text(strip=True)
            sex = sex_age[0] if sex_age else None
            birth_year = None
            if sex_age and len(sex_age) >= 2:
                try:
                    age = int(sex_age[1:])
                    # Calculate birth year from race date
                    if race.date:
                        race_year = int(race.date[:4])
                        birth_year = race_year - age
                except ValueError:
                    pass

            # Weight carried
            weight_carried = None
            try:
                weight_carried = float(cells[5].get_text(strip=True))
            except (ValueError, IndexError):
                pass

            # Jockey
            _, jockey_name_jp = _extract_jockey_name(cells[6])

            # Time
            time_secs = _parse_time(cells[7].get_text(strip=True))

            # Margin
            margin = cells[8].get_text(strip=True) if len(cells) > 8 else None

            # Odds & popularity
            odds_win = None
            popularity = None
            try:
                if len(cells) > 12:
                    odds_text = cells[12].get_text(strip=True)
                    # Handle empty, dashes, cancelled entries
                    if odds_text and odds_text not in ("", "---", "--", "-", "取消", "除外", "中止"):
                        odds_win = float(odds_text.replace(",", ""))
                    elif odds_text in ("取消", "除外", "中止"):
                        log.debug(f"Entry cancelled/excluded in race {race_id}: odds='{odds_text}'")
                if len(cells) > 13:
                    pop_text = cells[13].get_text(strip=True)
                    popularity = int(pop_text) if pop_text.isdigit() else None
            except (ValueError, IndexError):
                log.debug(f"Could not parse odds in race {race_id}: '{cells[12].get_text(strip=True) if len(cells) > 12 else '?'}'")
                pass

            # Horse weight
            horse_weight, horse_weight_change = None, None
            if len(cells) > 14:
                horse_weight, horse_weight_change = _parse_weight(cells[14].get_text(strip=True))

            # Last 3F
            last_3f = None
            if len(cells) > 11:
                last_3f = _parse_time(cells[11].get_text(strip=True))

            # Corner positions
            corners = None
            if len(cells) > 10:
                corners = cells[10].get_text(strip=True)

            horse = HorseData(
                name=horse_name_jp,  # will transliterate later
                name_jp=horse_name_jp,
                sex=sex,
                birth_year=birth_year,
                netkeiba_id=horse_id or f"unknown_{post_pos}",
            )

            entry = EntryData(
                post_position=post_pos or 0,
                draw=draw,
                horse=horse,
                jockey_name_jp=jockey_name_jp,
                weight_carried=weight_carried,
                horse_weight=horse_weight,
                horse_weight_change=horse_weight_change,
                odds_win=odds_win,
                popularity=popularity,
                finish_pos=finish_pos,
                margin=margin,
                time_secs=time_secs,
                last_3f_secs=last_3f,
                corner_positions=corners,
            )
            race.entries.append(entry)

        except Exception as e:
            log.warning(f"Error parsing row in race {race_id}: {e}")
            continue

    return race


# ---------------------------------------------------------------------------
# Database Insert
# ---------------------------------------------------------------------------

def save_race_to_db(race: RaceData) -> bool:
    """Insert a parsed race and all its entries/results into the database."""
    with get_session() as session:
        # Check if race already exists
        existing = session.execute(
            text("SELECT id FROM races WHERE netkeiba_id = :nid"),
            {"nid": race.netkeiba_id},
        ).fetchone()

        if existing:
            log.info(f"Race {race.netkeiba_id} already in DB (id={existing[0]}), skipping")
            return False

        # Upsert course
        course_id = None
        if race.course_code and race.course_code in JRA_COURSES:
            course_info = JRA_COURSES[race.course_code]
            result = session.execute(
                text("""
                    INSERT INTO courses (name, name_jp, surface, direction)
                    VALUES (:name, :name_jp, :surface, :direction)
                    ON CONFLICT DO NOTHING
                    RETURNING id
                """),
                course_info,
            ).fetchone()

            if result:
                course_id = result[0]
            else:
                course_id = session.execute(
                    text("SELECT id FROM courses WHERE name_jp = :name_jp"),
                    {"name_jp": course_info["name_jp"]},
                ).scalar()

        # Insert race
        race_result = session.execute(
            text("""
                INSERT INTO races (
                    netkeiba_id, date, course_id, race_number, distance,
                    surface, going, class, grade, race_name, race_name_jp,
                    weather, field_size
                ) VALUES (
                    :netkeiba_id, :date, :course_id, :race_number, :distance,
                    :surface, :going, :class, :grade, :race_name, :race_name_jp,
                    :weather, :field_size
                )
                RETURNING id
            """),
            {
                "netkeiba_id": race.netkeiba_id,
                "date": race.date,
                "course_id": course_id,
                "race_number": race.race_number,
                "distance": race.distance or 0,
                "surface": race.surface,
                "going": race.going,
                "class": race.class_,
                "grade": race.grade,
                "race_name": race.race_name,
                "race_name_jp": race.race_name_jp,
                "weather": race.weather,
                "field_size": race.field_size,
            },
        )
        race_db_id = race_result.fetchone()[0]

        # Insert entries + results
        for entry in race.entries:
            # Upsert horse
            horse = entry.horse
            horse_result = session.execute(
                text("""
                    INSERT INTO horses (name, name_jp, sex, birth_year, netkeiba_id)
                    VALUES (:name, :name_jp, :sex, :birth_year, :netkeiba_id)
                    ON CONFLICT (netkeiba_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        name_jp = EXCLUDED.name_jp
                    RETURNING id
                """),
                {
                    "name": horse.name,
                    "name_jp": horse.name_jp,
                    "sex": horse.sex,
                    "birth_year": horse.birth_year,
                    "netkeiba_id": horse.netkeiba_id,
                },
            )
            horse_db_id = horse_result.fetchone()[0]

            # Upsert jockey
            jockey_db_id = None
            if entry.jockey_name_jp:
                jockey_result = session.execute(
                    text("""
                        INSERT INTO jockeys (name, name_jp)
                        VALUES (:name, :name_jp)
                        ON CONFLICT DO NOTHING
                        RETURNING id
                    """),
                    {"name": entry.jockey_name_jp, "name_jp": entry.jockey_name_jp},
                ).fetchone()

                if jockey_result:
                    jockey_db_id = jockey_result[0]
                else:
                    jockey_db_id = session.execute(
                        text("SELECT id FROM jockeys WHERE name_jp = :name_jp"),
                        {"name_jp": entry.jockey_name_jp},
                    ).scalar()

            # Insert entry
            entry_result = session.execute(
                text("""
                    INSERT INTO entries (
                        race_id, horse_id, jockey_id, draw, post_position,
                        weight_carried, horse_weight, horse_weight_change,
                        odds_win, popularity
                    ) VALUES (
                        :race_id, :horse_id, :jockey_id, :draw, :post_position,
                        :weight_carried, :horse_weight, :horse_weight_change,
                        :odds_win, :popularity
                    )
                    ON CONFLICT (race_id, horse_id) DO NOTHING
                    RETURNING id
                """),
                {
                    "race_id": race_db_id,
                    "horse_id": horse_db_id,
                    "jockey_id": jockey_db_id,
                    "draw": entry.draw,
                    "post_position": entry.post_position,
                    "weight_carried": entry.weight_carried,
                    "horse_weight": entry.horse_weight,
                    "horse_weight_change": entry.horse_weight_change,
                    "odds_win": entry.odds_win,
                    "popularity": entry.popularity,
                },
            )
            entry_row = entry_result.fetchone()
            if not entry_row:
                continue
            entry_db_id = entry_row[0]

            # Insert result (if race is finished)
            if entry.finish_pos is not None or entry.time_secs is not None:
                session.execute(
                    text("""
                        INSERT INTO results (
                            entry_id, finish_pos, margin, time_secs,
                            last_3f_secs, corner_positions
                        ) VALUES (
                            :entry_id, :finish_pos, :margin, :time_secs,
                            :last_3f, :corners
                        )
                        ON CONFLICT (entry_id) DO NOTHING
                    """),
                    {
                        "entry_id": entry_db_id,
                        "finish_pos": entry.finish_pos,
                        "margin": entry.margin,
                        "time_secs": entry.time_secs,
                        "last_3f": entry.last_3f_secs,
                        "corners": entry.corner_positions,
                    },
                )

        log.info(
            f"✅ Saved race {race.netkeiba_id} "
            f"({race.race_name_jp}, {race.date}) — "
            f"{len(race.entries)} entries"
        )
        return True


# ---------------------------------------------------------------------------
# Scrape Orchestration
# ---------------------------------------------------------------------------

def scrape_race(race_id: str) -> Optional[RaceData]:
    """Scrape a single race by its netkeiba ID and save to DB."""
    log.info(f"Scraping race {race_id}...")
    url = RACE_URL.format(race_id=race_id)
    soup = _fetch(url)
    if not soup:
        return None

    race = parse_race_page(soup, race_id)
    if race and race.entries:
        save_race_to_db(race)
    else:
        log.warning(f"No entries parsed for race {race_id}")

    return race


def scrape_race_list(date_str: str) -> list[str]:
    """
    Scrape the race list for a given date and return netkeiba race IDs.
    Date format: YYYYMMDD
    """
    url = RACE_LIST_URL.format(date=date_str)
    log.info(f"Fetching race list for {date_str}...")
    soup = _fetch(url)
    if not soup:
        return []

    race_ids = []
    links = soup.find_all("a", href=re.compile(r"/race/\d{12}"))
    for link in links:
        match = re.search(r"/race/(\d{12})", link["href"])
        if match:
            rid = match.group(1)
            if rid not in race_ids:
                race_ids.append(rid)

    log.info(f"Found {len(race_ids)} races for {date_str}")
    return race_ids


def scrape_date(date_str: str) -> int:
    """Scrape all races for a given date. Returns count of new races saved."""
    race_ids = scrape_race_list(date_str)
    saved = 0
    for rid in race_ids:
        result = scrape_race(rid)
        if result:
            saved += 1
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Netkeiba Race Scraper")
    parser.add_argument("--race", type=str, help="Scrape a single race by netkeiba ID (12-digit)")
    parser.add_argument("--date", type=str, help="Scrape all races for a date (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument(
        "--year",
        type=int,
        help="Scrape all weekends for a given year (slow — use with --grades to filter)",
    )
    parser.add_argument(
        "--grades",
        type=str,
        default="G1,G2,G3",
        help="Comma-separated grades to filter (default: G1,G2,G3)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse only, don't save to DB")

    args = parser.parse_args()

    if args.race:
        scrape_race(args.race)
    elif args.date:
        date_str = args.date.replace("-", "")
        scrape_date(date_str)
    elif args.year:
        log.info(f"Year-based scraping for {args.year} — not yet implemented")
        log.info("Use --date to scrape specific race days for now")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
