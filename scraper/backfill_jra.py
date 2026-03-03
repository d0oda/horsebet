"""
UmaEdge — Backfill Races from JRA English Site.

Scrapes race results, pedigree (sires), trainers, and jockeys from the
official JRA English website (jra.jp/JRAEN). This bypasses netkeiba entirely.

The JRA EN site provides:
  - Full race results (FP, time, margin, odds)
  - Pedigree info (Sire, Dam, Dam's Sire, Dam's Dam)
  - Connections (Jockey, Trainer, Owner, Breeder)
  - Race metadata (distance, surface, class, weather, going)

API flow:
  1. POST /JRAEN/AP/kaisai/kaisaiRaceSearch  {raceYear}  → date list
  2. POST /JRAEN/AP/kaisai/raceSelect        {raceYmd}   → race list
  3. POST /JRAEN/AP/kaisai/running           {raceYmd, raceJoCd, raceKai, raceHi, raceNo} → results

Usage:
    python -m scraper.backfill_jra --year 2021                # full year
    python -m scraper.backfill_jra --year 2021 --month 6      # specific month
    python -m scraper.backfill_jra --date 20210109             # single date
    python -m scraper.backfill_jra --sires-only                # only update sires
    python -m scraper.backfill_jra --dry-run --year 2021       # preview
"""

import argparse
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

from scraper.db import get_session
from scraper.netkeiba import (
    HorseData,
    EntryData,
    RaceData,
    JRA_COURSES,
    save_race_to_db,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_jra")

BASE_URL = "https://jra.jp"
YEAR_SEARCH_URL = BASE_URL + "/JRAEN/AP/kaisai/kaisaiRaceSearch"
DATE_SELECT_URL = BASE_URL + "/JRAEN/AP/kaisai/raceSelect"
RACE_RESULT_URL = BASE_URL + "/JRAEN/AP/kaisai/running"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# JRA English venue name → netkeiba venue code mapping
VENUE_NAME_TO_CODE = {
    "SAPPORO": "01",
    "HAKODATE": "02",
    "FUKUSHIMA": "03",
    "NIIGATA": "04",
    "TOKYO": "05",
    "NAKAYAMA": "06",
    "CHUKYO": "07",
    "KYOTO": "08",
    "HANSHIN": "09",
    "KOKURA": "10",
}

# JRA English class abbreviations → Japanese class names
JRA_CLASS_MAP = {
    "MDN": "未勝利",    # Maiden
    "NWC": "1勝クラス",  # Newcomer / 1-win class
    "ALW": "オープン",   # Allowance → treat as open for compatibility
    "OPN": "オープン",   # Open
    "G1": "オープン",
    "G2": "オープン",
    "G3": "オープン",
    "L": "オープン",     # Listed
}

# JRA English sex codes
SEX_MAP = {
    "C": "牡",   # Colt
    "F": "牝",   # Filly
    "H": "セ",   # Horse (gelding)
    "M": "牝",   # Mare
    "R": "セ",   # Ridgling
    "G": "セ",   # Gelding
}


# ---------------------------------------------------------------------------
# HTTP Helpers
# ---------------------------------------------------------------------------

def _fetch_post(url: str, data: dict, retries: int = 3) -> Optional[str]:
    """POST to a JRA URL and return the response text."""
    for attempt in range(retries):
        try:
            time.sleep(1.0 + attempt * 0.5)  # polite delay
            resp = requests.post(url, headers=HEADERS, data=data, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 100:
                return resp.text
            elif resp.status_code == 200:
                log.debug(f"Empty response from {url}")
                return None
            else:
                log.warning(f"HTTP {resp.status_code} from {url} (attempt {attempt + 1})")
        except requests.RequestException as e:
            log.warning(f"Request error: {e} (attempt {attempt + 1})")
        time.sleep(2 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Discovery — Get Race Dates and Race Parameters
# ---------------------------------------------------------------------------

@dataclass
class JRARaceParams:
    """Parameters to fetch a single race result from JRA."""
    race_ymd: str      # YYYYMMDD
    race_jo_cd: str    # Venue code (01-10)
    race_kai: str      # Meeting number
    race_hi: str       # Day number
    race_no: str       # Race number (01-12)
    venue_name: str    # e.g. "NAKAYAMA"
    race_label: str    # e.g. "1R MDN D1200"


def get_race_dates(year: int) -> list[str]:
    """Get all race dates for a given year. Returns list of YYYYMMDD strings."""
    html = _fetch_post(YEAR_SEARCH_URL, {"raceYear": str(year)})
    if not html:
        log.error(f"Failed to fetch race dates for {year}")
        return []

    # Extract dates from JavaScript links like:
    # displayRace('SelectRace', '20210109')
    dates = re.findall(r"displayRace\('SelectRace',\s*'(\d{8})'\)", html)
    unique_dates = sorted(set(dates))
    log.info(f"Found {len(unique_dates)} race dates for {year}")
    return unique_dates


def get_races_for_date(race_ymd: str) -> list[JRARaceParams]:
    """Get all race parameters for a given date."""
    html = _fetch_post(DATE_SELECT_URL, {"raceYmd": race_ymd})
    if not html:
        log.error(f"Failed to fetch race list for {race_ymd}")
        return []

    # Extract race params from JavaScript links like:
    # displayRunning('SelectRunning', '20210109', '06', '01', '02', '01')
    pattern = r"displayRunning\('SelectRunning',\s*'(\d{8})',\s*'(\d{2})',\s*'(\d{2})',\s*'(\d{2})',\s*'(\d{2})'\)"
    matches = re.findall(pattern, html)

    # Also extract venue names and race labels from link text
    soup = BeautifulSoup(html, "lxml")
    all_links = soup.find_all("a")

    races = []
    seen = set()
    for link in all_links:
        href = link.get("href", "")
        m = re.search(pattern, href)
        if m:
            key = m.groups()
            if key in seen:
                continue
            seen.add(key)
            ymd, jo, kai, hi, no = key
            label = link.get_text(strip=True)

            # Find venue name by looking at parent context
            venue = ""
            for vname, vcode in VENUE_NAME_TO_CODE.items():
                if jo == vcode:
                    venue = vname
                    break

            races.append(JRARaceParams(
                race_ymd=ymd,
                race_jo_cd=jo,
                race_kai=kai,
                race_hi=hi,
                race_no=no,
                venue_name=venue,
                race_label=label,
            ))

    log.info(f"Found {len(races)} races for {race_ymd}")
    return races


# ---------------------------------------------------------------------------
# Parsing — Race Result Page
# ---------------------------------------------------------------------------

def _parse_time_en(time_str: str) -> Optional[float]:
    """Convert JRA English time string like '1:13.1' to seconds."""
    if not time_str:
        return None
    time_str = time_str.strip()
    try:
        if ":" in time_str:
            mins, secs = time_str.split(":")
            return float(mins) * 60 + float(secs)
        else:
            return float(time_str)
    except (ValueError, IndexError):
        return None


def _parse_margin_en(margin_str: str) -> Optional[str]:
    """Normalize JRA English margin string."""
    if not margin_str:
        return None
    margin_str = margin_str.strip()
    if margin_str in ("", "-", "--"):
        return None
    return margin_str


def parse_jra_result(html: str, params: JRARaceParams) -> Optional[RaceData]:
    """
    Parse a JRA English race result page into a RaceData object.

    The page has two main tables (class='beta'):
      1. Results table: FP, Bk, Hs, Horse, Sex/Age, Weight, Finish, Margin, Win Fav, Win Odds
      2. Pedigree table: FP, Horse, Sire/Dam, Dam's sire/Dam's dam, Jockey/Trainer, Owner/Breeder
    """
    soup = BeautifulSoup(html, "lxml")

    # Generate a synthetic netkeiba-style race ID
    # Format: YYYY + venue(2) + kai(2) + hi(2) + raceNo(2) = 12 digits
    year = params.race_ymd[:4]
    synthetic_id = f"{year}{params.race_jo_cd}{params.race_kai}{params.race_hi}{params.race_no}"

    race = RaceData(netkeiba_id=synthetic_id)
    race.course_code = params.race_jo_cd
    race.race_number = int(params.race_no)

    # Parse date from raceYmd
    ymd = params.race_ymd
    race.date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"

    # ---- Parse race name from plain tables ----
    # Table structure: tables[0] = venue nav, tables[1] = race name (e.g. "NAKAYAMA 1R"),
    # tables[2] = race conditions ("January 5, 2021, 1200m, Dirt Standard...")
    all_tables = soup.find_all("table")

    # Race name — in tables[1] text content
    if len(all_tables) > 1:
        race_name_text = all_tables[1].get_text(strip=True)
        if race_name_text:
            race.race_name = race_name_text
            race.race_name_jp = race_name_text

    # Race conditions — in tables[2] text content
    if len(all_tables) > 2:
        info_text = all_tables[2].get_text(strip=True)

        # Distance
        dist_match = re.search(r"(\d{3,4})m", info_text)
        if dist_match:
            race.distance = int(dist_match.group(1))

        # Surface
        if "Dirt" in info_text:
            race.surface = "dirt"
        elif "Turf" in info_text:
            race.surface = "turf"
        elif "Jump" in info_text or "Steeple" in info_text:
            race.surface = "turf"

        # Going
        going_map = {
            "Standard": "良",
            "Good": "稍重",
            "Yielding": "重",
            "Soft": "不良",
            "Heavy": "不良",
            "Firm": "良",
        }
        for en_going, jp_going in going_map.items():
            if en_going in info_text:
                race.going = jp_going
                break

        # Weather
        weather_map = {
            "Fine": "晴",
            "Cloudy": "曇",
            "Rainy": "雨",
            "Light Rain": "小雨",
            "Snowy": "雪",
        }
        for en_weather, jp_weather in weather_map.items():
            if en_weather in info_text:
                race.weather = jp_weather
                break

        # Post time
        post_match = re.search(r"Post time\s+(\d{2}:\d{2})", info_text)
        if post_match:
            race.post_time = post_match.group(1)

    # Parse class from the conditions line or race label
    # Look for patterns like "MIX DES, Weight for Age, 2-Year-Olds, Maiden"
    # or from the race_label like "1R MDN D1200"
    label = params.race_label
    for class_code, class_jp in JRA_CLASS_MAP.items():
        if class_code in label:
            race.class_ = class_jp
            break

    # Grade detection from label or page content
    grade_match = re.search(r"\(G([123I]+)\)", label + " " + (race.race_name or ""))
    if grade_match:
        grade_str = grade_match.group(1)
        if grade_str in ("1", "I"):
            race.grade = "G1"
        elif grade_str in ("2", "II"):
            race.grade = "G2"
        elif grade_str in ("3", "III"):
            race.grade = "G3"
        race.class_ = "オープン"
    elif "(L)" in label:
        race.grade = "L"
        race.class_ = "オープン"

    # Find result and pedigree tables using class='running'
    running_tables = soup.find_all("table", class_="running")
    # running_tables[0] = "Official" header table (FP/Bk/Hs cols)
    # running_tables[1] = actual result data table (with horse data)
    # running_tables[2] = lap times (class='running runningLap')
    # running_tables[3] = pedigree table (Sire/Dam/Jockey/Trainer)

    # Identify result data table and pedigree table by header text
    result_table = None
    pedigree_table = None
    for t in running_tables:
        header_row = t.find("tr")
        if header_row:
            header_text = header_row.get_text(strip=True)
            if "Horse" in header_text and "Finish" in header_text:
                result_table = t
            elif "Sire" in header_text and "Dam" in header_text:
                pedigree_table = t

    # ==== Parse result table ====
    # Table structure: Row 0 = header (FP,Bk,Hs,...), Row 1 = sub-header (1c,2c,3c,4c)
    # Data rows have 14 cells:
    #   [0]=FP, [1]=Bk, [2]=Hs, [3]=Horse, [4]=SexAge, [5]=Weight,
    #   [6-9]=Corner positions (1c,2c,3c,4c), [10]=Finish time,
    #   [11]=Margin, [12]=WinFav, [13]=WinOdds
    result_entries = {}  # keyed by post_position
    if result_table:
        all_rows = result_table.find_all("tr")
        # Skip header rows (< 10 cells), only parse data rows (14 cells)
        data_rows = [r for r in all_rows if len(r.find_all("td")) >= 10]
        race.field_size = len(data_rows)

        for row in data_rows:
            cells = row.find_all("td")
            try:
                # FP: "1st", "2nd", "3rd", "4th", etc.
                fp_text = cells[0].get_text(strip=True)
                finish_pos = None
                fp_match = re.match(r"(\d+)", fp_text)
                if fp_match:
                    finish_pos = int(fp_match.group(1))

                # Bk (bracket/draw) and Hs (horse number / post position)
                draw = None
                bk_text = cells[1].get_text(strip=True)
                draw = int(bk_text) if bk_text.isdigit() else None

                post_pos = None
                hs_text = cells[2].get_text(strip=True)
                post_pos = int(hs_text) if hs_text.isdigit() else None

                # Horse name: e.g. "Last Samurai(JPN)"
                horse_name_en = cells[3].get_text(strip=True)

                # Sex/Age: e.g. "C3" (Colt, 3 years old)
                sex_en = None
                birth_year = None
                sex_age_text = cells[4].get_text(strip=True)
                if sex_age_text and len(sex_age_text) >= 2:
                    sex_letter = sex_age_text[0]
                    sex_en = SEX_MAP.get(sex_letter, sex_letter)
                    try:
                        age = int(sex_age_text[1:])
                        race_year = int(params.race_ymd[:4])
                        birth_year = race_year - age
                    except ValueError:
                        pass

                # Weight carried (cell 5)
                weight_carried = None
                try:
                    weight_carried = float(cells[5].get_text(strip=True))
                except ValueError:
                    pass

                # Corner positions (cells 6-9)
                corners_parts = []
                for ci in range(6, min(10, len(cells))):
                    ct = cells[ci].get_text(strip=True)
                    if ct and ct.isdigit():
                        corners_parts.append(ct)
                corner_positions = "-".join(corners_parts) if corners_parts else None

                # Finish time (cell 10)
                time_secs = None
                if len(cells) > 10:
                    time_secs = _parse_time_en(cells[10].get_text(strip=True))

                # Margin (cell 11)
                margin = None
                if len(cells) > 11:
                    margin = _parse_margin_en(cells[11].get_text(strip=True))

                # Win Fav / popularity rank (cell 12)
                popularity = None
                if len(cells) > 12:
                    fav_text = cells[12].get_text(strip=True)
                    popularity = int(fav_text) if fav_text.isdigit() else None

                # Win Odds (cell 13)
                odds_win = None
                if len(cells) > 13:
                    odds_text = cells[13].get_text(strip=True)
                    try:
                        odds_win = float(odds_text)
                    except ValueError:
                        pass

                # Store entry by post position for pedigree matching
                key = post_pos or finish_pos or 0
                result_entries[key] = {
                    "finish_pos": finish_pos,
                    "draw": draw,
                    "post_pos": post_pos,
                    "horse_name_en": horse_name_en,
                    "sex": sex_en,
                    "birth_year": birth_year,
                    "weight_carried": weight_carried,
                    "time_secs": time_secs,
                    "margin": margin,
                    "popularity": popularity,
                    "odds_win": odds_win,
                    "corner_positions": corner_positions,
                }
            except Exception as e:
                log.warning(f"Error parsing result row: {e}")
                continue

    # ==== Parse pedigree table ====
    pedigree_data = {}  # keyed by horse_name_en
    if pedigree_table:
        ped_rows = pedigree_table.find_all("tr")[1:]  # skip header
        for row in ped_rows:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            try:
                # FP
                fp_text = cells[0].get_text(strip=True)

                # Horse name
                horse_name = cells[1].get_text(strip=True) if len(cells) > 1 else ""

                # Sire / Dam (in one cell, separated by <br>)
                sire_dam = ""
                sire_name = None
                if len(cells) > 2:
                    sire_dam_parts = cells[2].get_text(separator="|", strip=True).split("|")
                    if sire_dam_parts:
                        sire_name = sire_dam_parts[0].strip()

                # Jockey / Trainer (in one cell, separated by <br>)
                jockey_name = None
                trainer_name = None
                if len(cells) > 4:
                    jt_parts = cells[4].get_text(separator="|", strip=True).split("|")
                    if len(jt_parts) >= 1:
                        jockey_name = jt_parts[0].strip()
                    if len(jt_parts) >= 2:
                        trainer_name = jt_parts[1].strip()

                pedigree_data[horse_name] = {
                    "sire_name": sire_name,
                    "jockey_name": jockey_name,
                    "trainer_name": trainer_name,
                }
            except Exception as e:
                log.warning(f"Error parsing pedigree row: {e}")
                continue

    # ==== Combine results + pedigree into unified entries ====
    for key, entry_data in result_entries.items():
        horse_name_en = entry_data["horse_name_en"]

        # Look up pedigree
        ped = pedigree_data.get(horse_name_en, {})

        # Generate a synthetic horse ID based on name
        # (no netkeiba ID available from JRA site)
        safe_name = re.sub(r"[^a-zA-Z0-9]", "_", horse_name_en).lower()
        synthetic_horse_id = f"jra_{safe_name}"

        horse = HorseData(
            name=horse_name_en,
            name_jp=horse_name_en,  # English name — best we have
            sex=entry_data["sex"],
            birth_year=entry_data["birth_year"],
            netkeiba_id=synthetic_horse_id,
            sire_name=ped.get("sire_name"),
        )

        entry = EntryData(
            post_position=entry_data["post_pos"] or 0,
            draw=entry_data["draw"],
            horse=horse,
            jockey_name_jp=ped.get("jockey_name", ""),
            trainer_name_jp=ped.get("trainer_name", ""),
            weight_carried=entry_data["weight_carried"],
            odds_win=entry_data["odds_win"],
            popularity=entry_data["popularity"],
            finish_pos=entry_data["finish_pos"],
            margin=entry_data["margin"],
            time_secs=entry_data["time_secs"],
            corner_positions=entry_data.get("corner_positions"),
        )
        race.entries.append(entry)

    return race


# ---------------------------------------------------------------------------
# Sires-Only Mode — Update sires for existing races/horses
# ---------------------------------------------------------------------------

def backfill_sires_from_jra(year: int, month: Optional[int] = None,
                             dry_run: bool = False) -> dict:
    """
    For existing races in the DB, fetch pedigree data from JRA to fill
    missing sire_name. Matches by race date + post position.
    """
    stats = {"dates_checked": 0, "sires_updated": 0, "races_matched": 0}

    # Get existing race dates from DB
    with get_session() as session:
        query = """
            SELECT DISTINCT date::text FROM horsebet.races
            WHERE EXTRACT(YEAR FROM date) = :year
        """
        params = {"year": year}
        if month:
            query += " AND EXTRACT(MONTH FROM date) = :month"
            params["month"] = month
        query += " ORDER BY date"
        db_dates = [r[0] for r in session.execute(text(query), params).fetchall()]

    log.info(f"Found {len(db_dates)} existing race dates for {year}" +
             (f" month {month}" if month else ""))

    for date_str in db_dates:
        stats["dates_checked"] += 1
        ymd = date_str.replace("-", "")

        # Get race list from JRA
        races = get_races_for_date(ymd)
        if not races:
            continue

        for race_params in races:
            # Fetch the result page
            html = _fetch_post(RACE_RESULT_URL, {
                "raceYmd": race_params.race_ymd,
                "raceJoCd": race_params.race_jo_cd,
                "raceKai": race_params.race_kai,
                "raceHi": race_params.race_hi,
                "raceNo": race_params.race_no,
            })
            if not html:
                continue

            soup = BeautifulSoup(html, "lxml")
            running_tables = soup.find_all("table", class_="running")

            # Find result and pedigree tables by header content
            result_tbl = None
            ped_tbl = None
            for t in running_tables:
                header_row = t.find("tr")
                if header_row:
                    ht = header_row.get_text(strip=True)
                    if "Horse" in ht and "Finish" in ht:
                        result_tbl = t
                    elif "Sire" in ht and "Dam" in ht:
                        ped_tbl = t

            if not ped_tbl:
                continue

            # Parse result table to get post positions
            result_rows = [r for r in result_tbl.find_all("tr")
                          if len(r.find_all("td")) >= 10] if result_tbl else []

            # Parse pedigree table
            ped_rows = ped_tbl.find_all("tr")[1:]

            # Build name→sire map from pedigree
            name_sire = {}
            for row in ped_rows:
                cells = row.find_all("td")
                if len(cells) >= 3:
                    horse_name = cells[1].get_text(strip=True)
                    sire_parts = cells[2].get_text(separator="|", strip=True).split("|")
                    if sire_parts:
                        name_sire[horse_name] = sire_parts[0].strip()

            # Build post_pos→name map from result table
            pp_to_name = {}
            for row in result_rows:
                cells = row.find_all("td")
                if len(cells) >= 4:
                    hs_text = cells[2].get_text(strip=True)
                    horse_name = cells[3].get_text(strip=True)
                    if hs_text.isdigit():
                        pp_to_name[int(hs_text)] = horse_name

            # Match with DB entries by date + venue + race_no + post_position
            venue_code = race_params.race_jo_cd
            race_no = int(race_params.race_no)

            if not dry_run:
                with get_session() as session:
                    # Find matching race in DB
                    db_race = session.execute(text("""
                        SELECT r.id FROM horsebet.races r
                        JOIN horsebet.courses c ON c.id = r.course_id
                        WHERE r.date = :date AND r.race_number = :rnum
                          AND c.name_jp = :venue_jp
                    """), {
                        "date": date_str,
                        "rnum": race_no,
                        "venue_jp": JRA_COURSES.get(venue_code, {}).get("name_jp", ""),
                    }).fetchone()

                    if not db_race:
                        continue

                    race_db_id = db_race[0]
                    stats["races_matched"] += 1

                    # Update sires for each horse
                    for pp, horse_name_en in pp_to_name.items():
                        sire = name_sire.get(horse_name_en)
                        if not sire:
                            continue

                        result = session.execute(text("""
                            UPDATE horsebet.horses h
                            SET sire_name = :sire
                            FROM horsebet.entries e
                            WHERE e.horse_id = h.id
                              AND e.race_id = :race_id
                              AND e.post_position = :pp
                              AND h.sire_name IS NULL
                        """), {
                            "sire": sire,
                            "race_id": race_db_id,
                            "pp": pp,
                        })
                        if result.rowcount > 0:
                            stats["sires_updated"] += result.rowcount
            else:
                for pp, horse_name_en in pp_to_name.items():
                    sire = name_sire.get(horse_name_en)
                    if sire:
                        log.debug(f"  Would update: pp={pp} {horse_name_en} → sire={sire}")

        if stats["dates_checked"] % 5 == 0:
            log.info(f"  Progress: {stats['dates_checked']}/{len(db_dates)} dates, "
                     f"+{stats['sires_updated']} sires, {stats['races_matched']} races matched")

    return stats


# ---------------------------------------------------------------------------
# Full Scrape — Discover and insert new races
# ---------------------------------------------------------------------------

# Thread-safe counter
_stats_lock = threading.Lock()


def _scrape_one_race(race_params: JRARaceParams) -> dict:
    """Fetch, parse, and save a single race. Returns stats dict."""
    result = {"saved": 0, "skipped": 0, "sires": 0, "error": 0}

    year_str = race_params.race_ymd[:4]
    synthetic_id = (f"{year_str}{race_params.race_jo_cd}"
                    f"{race_params.race_kai}{race_params.race_hi}"
                    f"{race_params.race_no}")

    # Check if already exists
    with get_session() as session:
        existing = session.execute(
            text("SELECT id FROM races WHERE netkeiba_id = :nid"),
            {"nid": synthetic_id},
        ).fetchone()
        if existing:
            result["skipped"] = 1
            return result

    # Fetch and parse
    html = _fetch_post(RACE_RESULT_URL, {
        "raceYmd": race_params.race_ymd,
        "raceJoCd": race_params.race_jo_cd,
        "raceKai": race_params.race_kai,
        "raceHi": race_params.race_hi,
        "raceNo": race_params.race_no,
    })
    if not html:
        result["error"] = 1
        return result

    race = parse_jra_result(html, race_params)
    if not race or not race.entries:
        result["error"] = 1
        return result

    result["sires"] = sum(1 for e in race.entries if e.horse.sire_name)

    try:
        save_race_to_db(race)
        result["saved"] = 1
    except Exception as e:
        err_str = str(e)
        if "UniqueViolation" in err_str or "duplicate key" in err_str:
            result["skipped"] = 1  # another worker beat us
        else:
            log.warning(f"  Error saving {synthetic_id}: {e}")
            result["error"] = 1

    return result


def _scrape_date_batch(ymd: str) -> dict:
    """Discover and scrape all races for a single date. Returns stats."""
    batch = {"saved": 0, "skipped": 0, "sires": 0, "errors": 0}

    race_list = get_races_for_date(ymd)
    if not race_list:
        return batch

    for race_params in race_list:
        r = _scrape_one_race(race_params)
        batch["saved"] += r["saved"]
        batch["skipped"] += r["skipped"]
        batch["sires"] += r["sires"]
        batch["errors"] += r["error"]

    date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    if batch["saved"] > 0:
        log.info(f"  ✅ {date_str}: +{batch['saved']} races, "
                 f"+{batch['sires']} sires "
                 f"({batch['skipped']} skipped)")

    return batch


def scrape_year(year: int, month: Optional[int] = None,
                dry_run: bool = False, workers: int = 4) -> dict:
    """Scrape all races for a year (or month) from JRA and insert into DB."""
    stats = {"dates_checked": 0, "dates_with_races": 0,
             "races_saved": 0, "races_skipped": 0, "sires_added": 0}

    # Get all dates for the year
    dates = get_race_dates(year)
    if month:
        month_str = f"{year}{month:02d}"
        dates = [d for d in dates if d.startswith(month_str)]
        log.info(f"Filtered to {len(dates)} dates for {year}/{month:02d}")

    if dry_run:
        for date_idx, ymd in enumerate(dates, 1):
            race_list = get_races_for_date(ymd)
            date_str = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
            for rp in race_list:
                log.info(f"  [{date_idx}/{len(dates)}] Would scrape {date_str} "
                         f"{rp.venue_name} {rp.race_label}")
                stats["races_saved"] += 1
            stats["dates_checked"] += 1
            stats["dates_with_races"] += 1
        return stats

    log.info(f"Scraping {len(dates)} dates with {workers} workers...")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scrape_date_batch, ymd): ymd for ymd in dates}
        completed = 0

        for future in as_completed(futures):
            ymd = futures[future]
            completed += 1
            try:
                batch = future.result()
                stats["dates_checked"] += 1
                if batch["saved"] > 0:
                    stats["dates_with_races"] += 1
                stats["races_saved"] += batch["saved"]
                stats["races_skipped"] += batch["skipped"]
                stats["sires_added"] += batch["sires"]
            except Exception as e:
                log.warning(f"  Date {ymd} failed: {e}")

            if completed % 10 == 0 or completed == len(dates):
                log.info(f"  Progress: {completed}/{len(dates)} dates | "
                         f"+{stats['races_saved']} races | "
                         f"{stats['races_skipped']} skipped | "
                         f"+{stats['sires_added']} sires")

    return stats


def scrape_single_date(ymd: str, dry_run: bool = False) -> dict:
    """Scrape all races for a single date."""
    stats = {"races_saved": 0, "races_skipped": 0, "sires_added": 0}

    race_list = get_races_for_date(ymd)
    if not race_list:
        log.error(f"No races found for {ymd}")
        return stats

    for race_params in race_list:
        if dry_run:
            log.info(f"  Would scrape {race_params.venue_name} {race_params.race_label}")
            stats["races_saved"] += 1
            continue

        html = _fetch_post(RACE_RESULT_URL, {
            "raceYmd": race_params.race_ymd,
            "raceJoCd": race_params.race_jo_cd,
            "raceKai": race_params.race_kai,
            "raceHi": race_params.race_hi,
            "raceNo": race_params.race_no,
        })
        if not html:
            continue

        race = parse_jra_result(html, race_params)
        if not race or not race.entries:
            continue

        sires = sum(1 for e in race.entries if e.horse.sire_name)
        stats["sires_added"] += sires

        try:
            save_race_to_db(race)
            stats["races_saved"] += 1
        except Exception as e:
            log.warning(f"  Error: {e}")
            stats["races_skipped"] += 1

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UmaEdge — Backfill from JRA English Site",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--year", type=int, help="Scrape a full year")
    parser.add_argument("--month", type=int, help="With --year, limit to a month")
    parser.add_argument("--date", type=str, help="Single date (YYYYMMDD)")
    parser.add_argument("--sires-only", action="store_true",
                        help="Only update sires for existing races")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview mode, no DB writes")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent workers (default: 4)")

    args = parser.parse_args()

    if args.sires_only:
        if not args.year:
            log.error("--sires-only requires --year")
            return
        log.info(f"Backfilling sires from JRA for {args.year}"
                 + (f" month {args.month}" if args.month else ""))
        stats = backfill_sires_from_jra(args.year, args.month, args.dry_run)
        log.info(f"\n{'=' * 50}")
        log.info(f"Sire backfill complete:")
        log.info(f"  Dates checked: {stats['dates_checked']}")
        log.info(f"  Races matched: {stats['races_matched']}")
        log.info(f"  Sires updated: {stats['sires_updated']}")
        if args.dry_run:
            log.info("(dry run — no DB changes)")
        log.info(f"{'=' * 50}")

    elif args.year:
        log.info(f"Scraping {args.year}" +
                 (f" month {args.month}" if args.month else ""))
        stats = scrape_year(args.year, args.month, args.dry_run, args.workers)
        log.info(f"\n{'=' * 50}")
        log.info(f"Scrape complete:")
        log.info(f"  Dates checked: {stats['dates_checked']}")
        log.info(f"  Dates with races: {stats['dates_with_races']}")
        log.info(f"  Races saved: {stats['races_saved']}")
        log.info(f"  Races skipped: {stats['races_skipped']}")
        log.info(f"  Sires added: {stats['sires_added']}")
        if args.dry_run:
            log.info("(dry run — no DB changes)")
        log.info(f"{'=' * 50}")

    elif args.date:
        log.info(f"Scraping single date: {args.date}")
        stats = scrape_single_date(args.date, args.dry_run)
        log.info(f"\n{'=' * 50}")
        log.info(f"Complete: +{stats['races_saved']} races, "
                 f"+{stats['sires_added']} sires")
        if args.dry_run:
            log.info("(dry run — no DB changes)")
        log.info(f"{'=' * 50}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
