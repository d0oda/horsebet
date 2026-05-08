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

# race.netkeiba.com — the live site (db.netkeiba.com returns 400 as of Feb 2026)
BASE_URL = "https://race.netkeiba.com"
RESULT_URL = BASE_URL + "/race/result.html?race_id={race_id}"
SHUTUBA_URL = BASE_URL + "/race/shutuba.html?race_id={race_id}"
HORSE_URL = "https://db.netkeiba.com/horse/{horse_id}/"
RACE_LIST_URL = BASE_URL + "/top/race_list.html?kaisai_date={date}"

# Legacy db.netkeiba.com URL (kept for backward compat, but returns 400 now)
LEGACY_RACE_URL = "https://db.netkeiba.com/race/{race_id}/"

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
    trainer_name_jp: Optional[str] = None  # 調教師
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
    post_time: Optional[str] = None       # HH:MM JST
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
            resp = requests.get(url, headers=HEADERS, timeout=(10.0, 20.0))
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
# Parsing — Race Class Extraction
# ---------------------------------------------------------------------------

def _extract_race_class(race_name_jp: Optional[str], grade: Optional[str]) -> Optional[str]:
    """
    Extract race class from the Japanese race name.

    Patterns:
        '東京優駿(G1)'     → 'G1'   (via grade)
        'クローバー賞(OP)' → 'OP'
        'STV賞(3勝)'      → '3勝'
        '積丹特別(2勝)'    → '2勝'
        'ニセコ特別(1勝)'  → '1勝'
        'BSN賞(L)'         → 'L'
        '3歳未勝利'        → '未勝利'
        '2歳新馬'          → '新馬'
        '障害4歳以上オープン' → 'OP'
        '3歳以上1勝クラス'  → '1勝'
    """
    # If a grade was already detected, use it
    if grade:
        return grade

    if not race_name_jp:
        return None

    name = race_name_jp

    # Parenthesised class: e.g. '(OP)', '(3勝)', '(L)'
    paren_match = re.search(r'[（(](G[1-3I]+|OP|L|[1-3]勝)[）)]', name)
    if paren_match:
        cls = paren_match.group(1)
        # Normalise GI→G1 etc.
        cls = cls.replace('GI', 'G1').replace('GII', 'G2').replace('GIII', 'G3')
        return cls

    # Direct patterns in the name
    if '新馬' in name:
        return '新馬'
    if '未勝利' in name:
        return '未勝利'
    if 'オープン' in name:
        return 'OP'
    # e.g. '3歳以上1勝クラス'
    class_match = re.search(r'([1-3]勝)クラス', name)
    if class_match:
        return class_match.group(1)

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


def _parse_race_header(soup: BeautifulSoup, race_id: str) -> RaceData:
    """
    Parse race header info common to both shutuba and result pages
    on race.netkeiba.com (uses .RaceName, .RaceData01 selectors).
    """
    race = RaceData(netkeiba_id=race_id)

    # Race name
    rname_el = soup.select_one(".RaceName")
    if rname_el:
        race.race_name_jp = rname_el.get_text(strip=True)
        race.race_name = race.race_name_jp

    # Race details (distance, surface, going, weather)
    rd01 = soup.select_one(".RaceData01")
    if rd01:
        detail_text = rd01.get_text()
        # Distance + surface: "ダ1400m" or "芝2000m" or "障3000m"
        m = re.search(r"(ダ|芝|障)(\d{3,4})m", detail_text)
        if m:
            race.surface = "dirt" if m.group(1) == "ダ" else "turf"
            race.distance = int(m.group(2))
        # Going
        going_match = re.search(r"馬場[：:]\s*(良|稍重|稍|不良|不|重)", detail_text)
        if going_match:
            going_val = going_match.group(1)
            # Normalize abbreviated forms
            if going_val == "稍":
                going_val = "稍重"
            elif going_val == "不":
                going_val = "不良"
            race.going = going_val
        else:
            # Also check spans for going
            for span in rd01.select("span"):
                t = span.get_text(strip=True)
                if t in ("良", "稍重", "稍", "重", "不良", "不"):
                    race.going = {"稍": "稍重", "不": "不良"}.get(t, t)
                    break
        # Weather
        weather_match = re.search(r"天候[：:]\s*(\S+)", detail_text)
        if weather_match:
            race.weather = weather_match.group(1)
        # Post time (発走 HH:MM or just HH:MM)
        time_match = re.search(r'(\d{1,2}:\d{2})', detail_text)
        if time_match:
            race.post_time = time_match.group(1)

    # Date and course from race ID (format: YYYYCCDDRRNN)
    if len(race_id) >= 12:
        race.course_code = race_id[4:6]
        try:
            race.race_number = int(race_id[10:12])
        except (ValueError, IndexError):
            pass

    # Try to get date from RaceData02, page title, or meta tags
    rd02 = soup.select_one(".RaceData02")
    if rd02:
        date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", rd02.get_text())
        if date_match:
            race.date = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"

    if not race.date:
        # Fallback: check page <title> or meta description for date
        title_tag = soup.find("title")
        if title_tag:
            date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title_tag.get_text())
            if date_match:
                race.date = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"

    if not race.date:
        # Last fallback: try meta og:description
        meta_desc = soup.find("meta", attrs={"property": "og:description"})
        if meta_desc:
            content = meta_desc.get("content", "")
            date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", content)
            if date_match:
                race.date = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"

    # Grade detection
    name_text = race.race_name_jp or ""
    for g in ["G1", "G2", "G3", "GI", "GII", "GIII", "Ｇ１", "Ｇ２", "Ｇ３"]:
        if g in name_text:
            race.grade = g.replace("Ｇ", "G").replace("I", "1").replace("II", "2").replace("III", "3")
            break
    race.class_ = _extract_race_class(race.race_name_jp, race.grade)

    return race


def _parse_odds_text(text: str) -> Optional[float]:
    """Parse odds text, handling dashes and cancelled entries."""
    if not text:
        return None
    text = text.strip().replace(",", "")
    if text in ("", "---", "---.-", "--", "-", "**", "取消", "除外", "中止"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_shutuba_page(soup: BeautifulSoup, race_id: str) -> Optional[RaceData]:
    """
    Parse a race.netkeiba.com SHUTUBA (pre-race / entry list) page.

    Cell layout for tr.HorseList rows (15 cells):
        0: Waku (枠番/draw)         — e.g. "1"
        1: Umaban (馬番/post pos)    — e.g. "1"
        2: CheckMark (prediction)    — skip
        3: HorseInfo (name, ID)      — contains span.HorseName > a
        4: Barei (sex+age)           — e.g. "牝3"
        5: Weight carried            — e.g. "55.0"
        6: Jockey                    — link text
        7: Trainer                   — link text
        8: Horse weight              — e.g. "488(+4)"
        9: Odds (Txt_R Popular)      — e.g. "10.5" or "---.-"
       10: Popularity rank           — e.g. "5" or "**"
       11-14: FavRegist, Memo, Notes — skip
    """
    race = _parse_race_header(soup, race_id)
    if not race.race_name_jp:
        log.warning(f"No race header found for shutuba {race_id}")
        return None

    rows = soup.select("tr.HorseList")
    if not rows:
        log.warning(f"No HorseList rows in shutuba {race_id}")
        return race

    race.field_size = len(rows)

    for row in rows:
        cells = row.select("td")
        if len(cells) < 9:
            continue

        try:
            # Draw (枠番) and post position (馬番)
            draw_text = cells[0].get_text(strip=True)
            draw = int(draw_text) if draw_text.isdigit() else None

            pp_text = cells[1].get_text(strip=True)
            post_pos = int(pp_text) if pp_text.isdigit() else None

            # Horse name + ID
            horse_link = row.select_one("span.HorseName a") or row.select_one("a[href*='/horse/']")
            horse_name_jp = horse_link.get_text(strip=True) if horse_link else cells[3].get_text(strip=True)
            horse_id = None
            if horse_link and horse_link.get("href"):
                horse_id = _extract_horse_id(horse_link["href"])

            # Sex and age
            sex_age = cells[4].get_text(strip=True)
            sex = sex_age[0] if sex_age else None
            birth_year = None
            if sex_age and len(sex_age) >= 2:
                try:
                    age = int(sex_age[1:])
                    if race.date:
                        birth_year = int(race.date[:4]) - age
                except ValueError:
                    pass

            # Weight carried
            weight_carried = None
            try:
                weight_carried = float(cells[5].get_text(strip=True))
            except (ValueError, IndexError):
                pass

            # Jockey
            jockey_cell = cells[6]
            jockey_link = jockey_cell.find("a")
            jockey_name_jp = jockey_link.get_text(strip=True) if jockey_link else jockey_cell.get_text(strip=True)

            # Trainer
            trainer_name_jp = None
            trainer_cell = cells[7]
            trainer_link = trainer_cell.find("a")
            if trainer_link:
                trainer_name_jp = trainer_link.get_text(strip=True)
            else:
                # Trainer text often includes stable prefix (e.g. "栗東小崎")
                trainer_text = trainer_cell.get_text(strip=True)
                if trainer_text:
                    # Strip stable prefix (美浦/栗東)
                    trainer_name_jp = re.sub(r'^(美浦|栗東)', '', trainer_text) or trainer_text

            # Horse weight
            horse_weight, horse_weight_change = None, None
            if len(cells) > 8:
                horse_weight, horse_weight_change = _parse_weight(cells[8].get_text(strip=True))

            # Odds
            odds_win = None
            if len(cells) > 9:
                odds_win = _parse_odds_text(cells[9].get_text(strip=True))

            # Popularity
            popularity = None
            if len(cells) > 10:
                pop_text = cells[10].get_text(strip=True)
                popularity = int(pop_text) if pop_text.isdigit() else None

            horse = HorseData(
                name=horse_name_jp,
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
                trainer_name_jp=trainer_name_jp,
                weight_carried=weight_carried,
                horse_weight=horse_weight,
                horse_weight_change=horse_weight_change,
                odds_win=odds_win,
                popularity=popularity,
            )
            race.entries.append(entry)

        except Exception as e:
            log.warning(f"Error parsing shutuba row in {race_id}: {e}")
            continue

    return race


def parse_result_page(soup: BeautifulSoup, race_id: str) -> Optional[RaceData]:
    """
    Parse a race.netkeiba.com RESULT page (finished race).

    Cell layout for tr.HorseList rows (15 cells):
        0: Result_Num (finish position)
        1: Waku (枠番/draw)        — class Num WakuN
        2: Umaban (馬番/post pos)   — class Num Txt_C
        3: Horse_Info (name, ID)    — contains span.Horse_Name > a (note underscore)
        4: Horse_Info (sex+age)
        5: Jockey_Info (weight carried)
        6: Jockey
        7: Time
        8: Margin
        9: Odds popularity (rank)   — class Odds Txt_C
       10: Odds (win odds)          — class Odds Txt_R
       11: Last 3F                  — class Time BgOrange
       12: PassageRate (corners)    — e.g. "1-1-1-1"
       13: Trainer
       14: Weight                   — horse weight
    """
    race = _parse_race_header(soup, race_id)
    if not race.race_name_jp:
        log.warning(f"No race header found for result {race_id}")
        return None

    rows = soup.select("tr.HorseList")
    if not rows:
        log.warning(f"No HorseList rows in result {race_id}")
        return race

    race.field_size = len(rows)

    for row in rows:
        cells = row.select("td")
        if len(cells) < 11:
            continue

        try:
            # Finish position
            finish_text = cells[0].get_text(strip=True)
            finish_pos = int(finish_text) if finish_text.isdigit() else None

            # Draw (枠番) and post position (馬番)
            draw_text = cells[1].get_text(strip=True)
            draw = int(draw_text) if draw_text.isdigit() else None

            pp_text = cells[2].get_text(strip=True)
            post_pos = int(pp_text) if pp_text.isdigit() else None

            # Horse name + ID
            horse_link = (
                row.select_one("span.Horse_Name a")
                or row.select_one("span.HorseName a")
                or row.select_one("a[href*='/horse/']")
            )
            horse_name_jp = horse_link.get_text(strip=True) if horse_link else cells[3].get_text(strip=True)
            horse_id = None
            if horse_link and horse_link.get("href"):
                horse_id = _extract_horse_id(horse_link["href"])

            # Sex and age
            sex_age = cells[4].get_text(strip=True)
            sex = sex_age[0] if sex_age else None
            birth_year = None
            if sex_age and len(sex_age) >= 2:
                try:
                    age = int(sex_age[1:])
                    if race.date:
                        birth_year = int(race.date[:4]) - age
                except ValueError:
                    pass

            # Weight carried
            weight_carried = None
            try:
                weight_carried = float(cells[5].get_text(strip=True))
            except (ValueError, IndexError):
                pass

            # Jockey
            jockey_cell = cells[6]
            jockey_link = jockey_cell.find("a")
            jockey_name_jp = jockey_link.get_text(strip=True) if jockey_link else jockey_cell.get_text(strip=True)

            # Time
            time_secs = _parse_time(cells[7].get_text(strip=True))

            # Margin
            margin = cells[8].get_text(strip=True) if len(cells) > 8 else None

            # Popularity (rank)
            popularity = None
            if len(cells) > 9:
                pop_text = cells[9].get_text(strip=True)
                popularity = int(pop_text) if pop_text.isdigit() else None

            # Odds
            odds_win = None
            if len(cells) > 10:
                odds_win = _parse_odds_text(cells[10].get_text(strip=True))

            # Last 3F
            last_3f = None
            if len(cells) > 11:
                last_3f = _parse_time(cells[11].get_text(strip=True))

            # Corner positions
            corners = None
            if len(cells) > 12:
                corners = cells[12].get_text(strip=True)

            # Trainer
            trainer_name_jp = None
            if len(cells) > 13:
                trainer_cell = cells[13]
                trainer_link = trainer_cell.find("a")
                if trainer_link:
                    trainer_name_jp = trainer_link.get_text(strip=True)
                else:
                    trainer_text = trainer_cell.get_text(strip=True)
                    if trainer_text:
                        trainer_name_jp = re.sub(r'^(美浦|栗東)', '', trainer_text) or trainer_text

            # Horse weight
            horse_weight, horse_weight_change = None, None
            if len(cells) > 14:
                horse_weight, horse_weight_change = _parse_weight(cells[14].get_text(strip=True))

            horse = HorseData(
                name=horse_name_jp,
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
                trainer_name_jp=trainer_name_jp,
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
            log.warning(f"Error parsing result row in {race_id}: {e}")
            continue

    return race


def parse_race_page(soup: BeautifulSoup, race_id: str) -> Optional[RaceData]:
    """
    Parse a netkeiba race page — auto-detects result vs shutuba format.
    Falls back to legacy db.netkeiba.com format if neither matches.
    """
    # Detect page type by checking for result-specific markers
    has_result_num = soup.select_one("td.Result_Num") is not None
    has_horse_list = bool(soup.select("tr.HorseList"))

    if has_horse_list:
        if has_result_num:
            return parse_result_page(soup, race_id)
        else:
            return parse_shutuba_page(soup, race_id)

    # Legacy fallback for db.netkeiba.com format (race_table_01)
    log.debug(f"Trying legacy parser for {race_id}")
    return _parse_legacy_race_page(soup, race_id)


def _parse_legacy_race_page(soup: BeautifulSoup, race_id: str) -> Optional[RaceData]:
    """Parse a legacy db.netkeiba.com race result page (race_table_01 format)."""
    race = RaceData(netkeiba_id=race_id)

    header = soup.find("div", class_="data_intro")
    if not header:
        log.warning(f"No race header found for {race_id} (legacy parser)")
        return None

    title = header.find("h1")
    if title:
        race.race_name_jp = title.get_text(strip=True)
        race.race_name = race.race_name_jp

    detail_span = header.find("span")
    if detail_span:
        detail_text = detail_span.get_text(strip=True)
        dist_match = re.search(r"(\d{3,4})m", detail_text)
        if dist_match:
            race.distance = int(dist_match.group(1))
        if "ダート" in detail_text or "ダ" in detail_text:
            race.surface = "dirt"
        elif "芝" in detail_text:
            race.surface = "turf"
        going_match = re.search(r"[芝ダート]+\s*[：:]\s*(良|稍重|稍|不良|不|重)", detail_text)
        if not going_match:
            # Broader fallback: 馬場:良 format
            going_match = re.search(r"馬場[：:]\s*(良|稍重|稍|不良|不|重)", detail_text)
        if going_match:
            going_val = going_match.group(1)
            if going_val == "稍":
                going_val = "稍重"
            elif going_val == "不":
                going_val = "不良"
            race.going = going_val
        weather_match = re.search(r"天候\s*[：:]\s*(\S+)", detail_text)
        if weather_match:
            race.weather = weather_match.group(1)

    # Extract actual date from page — the data_intro section contains YYYY年M月D日
    # NOTE: Do NOT derive date from race_id bytes 4-8 — those are venue+meeting codes,
    # NOT month+day!  Race ID format: YYYY-VV-WW-DD-RR (venue, meeting, day-in-meeting, race)
    if not race.date:
        page_text = header.get_text() if header else ""
        date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", page_text)
        if date_match:
            race.date = f"{date_match.group(1)}-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}"

    if len(race_id) >= 12:
        race.course_code = race_id[4:6]
        try:
            race.race_number = int(race_id[10:12])
        except (ValueError, IndexError):
            pass

    name_text = race.race_name_jp or ""
    for g in ["G1", "G2", "G3", "GI", "GII", "GIII", "Ｇ１", "Ｇ２", "Ｇ３"]:
        if g in name_text:
            race.grade = g.replace("Ｇ", "G").replace("I", "1").replace("II", "2").replace("III", "3")
            break
    race.class_ = _extract_race_class(race.race_name_jp, race.grade)

    result_table = soup.find("table", class_="race_table_01")
    if not result_table:
        log.warning(f"No result table found for {race_id}")
        return race

    rows = result_table.find_all("tr")[1:]
    race.field_size = len(rows)

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 13:
            continue
        try:
            finish_text = cells[0].get_text(strip=True)
            finish_pos = int(finish_text) if finish_text.isdigit() else None
            draw = int(cells[1].get_text(strip=True)) if cells[1].get_text(strip=True).isdigit() else None
            post_pos = int(cells[2].get_text(strip=True)) if cells[2].get_text(strip=True).isdigit() else None
            horse_cell = cells[3]
            horse_link = horse_cell.find("a")
            horse_name_jp = horse_link.get_text(strip=True) if horse_link else cells[3].get_text(strip=True)
            horse_id = _extract_horse_id(horse_link["href"]) if horse_link and horse_link.get("href") else None
            sex_age = cells[4].get_text(strip=True)
            sex = sex_age[0] if sex_age else None
            birth_year = None
            if sex_age and len(sex_age) >= 2:
                try:
                    age = int(sex_age[1:])
                    if race.date:
                        birth_year = int(race.date[:4]) - age
                except ValueError:
                    pass
            weight_carried = None
            try:
                weight_carried = float(cells[5].get_text(strip=True))
            except (ValueError, IndexError):
                pass
            _, jockey_name_jp = _extract_jockey_name(cells[6])
            time_secs = _parse_time(cells[7].get_text(strip=True))
            margin = cells[8].get_text(strip=True) if len(cells) > 8 else None
            odds_win = _parse_odds_text(cells[12].get_text(strip=True)) if len(cells) > 12 else None
            popularity = None
            if len(cells) > 13:
                pop_text = cells[13].get_text(strip=True)
                popularity = int(pop_text) if pop_text.isdigit() else None
            horse_weight, horse_weight_change = None, None
            if len(cells) > 14:
                horse_weight, horse_weight_change = _parse_weight(cells[14].get_text(strip=True))
            last_3f = _parse_time(cells[11].get_text(strip=True)) if len(cells) > 11 else None
            corners = cells[10].get_text(strip=True) if len(cells) > 10 else None
            trainer_name_jp = None
            for ci in [18, 19, 17, 16, 15]:
                if ci < len(cells):
                    trainer_link = cells[ci].find("a", href=re.compile(r'/trainer/'))
                    if trainer_link:
                        trainer_name_jp = trainer_link.get_text(strip=True)
                        break

            horse = HorseData(
                name=horse_name_jp, name_jp=horse_name_jp, sex=sex,
                birth_year=birth_year, netkeiba_id=horse_id or f"unknown_{post_pos}",
            )
            entry = EntryData(
                post_position=post_pos or 0, draw=draw, horse=horse,
                jockey_name_jp=jockey_name_jp, trainer_name_jp=trainer_name_jp,
                weight_carried=weight_carried, horse_weight=horse_weight,
                horse_weight_change=horse_weight_change, odds_win=odds_win,
                popularity=popularity, finish_pos=finish_pos, margin=margin,
                time_secs=time_secs, last_3f_secs=last_3f, corner_positions=corners,
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
                    ON CONFLICT (name_jp) DO NOTHING
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
                    weather, field_size, post_time
                ) VALUES (
                    :netkeiba_id, :date, :course_id, :race_number, :distance,
                    :surface, :going, :class, :grade, :race_name, :race_name_jp,
                    :weather, :field_size, :post_time
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
                "post_time": race.post_time,
            },
        )
        race_db_id = race_result.fetchone()[0]

        # Insert entries + results
        for entry in race.entries:
            # Upsert trainer
            trainer_db_id = None
            if entry.trainer_name_jp:
                trainer_result = session.execute(
                    text("""
                        INSERT INTO trainers (name, name_jp)
                        VALUES (:name, :name_jp)
                        ON CONFLICT (name_jp) DO NOTHING
                        RETURNING id
                    """),
                    {"name": entry.trainer_name_jp, "name_jp": entry.trainer_name_jp},
                ).fetchone()

                if trainer_result:
                    trainer_db_id = trainer_result[0]
                else:
                    trainer_db_id = session.execute(
                        text("SELECT id FROM trainers WHERE name_jp = :name_jp"),
                        {"name_jp": entry.trainer_name_jp},
                    ).scalar()

            # Upsert horse (with trainer_id)
            horse = entry.horse
            horse_result = session.execute(
                text("""
                    INSERT INTO horses (name, name_jp, sex, birth_year, netkeiba_id, sire_name, trainer_id)
                    VALUES (:name, :name_jp, :sex, :birth_year, :netkeiba_id, :sire_name, :trainer_id)
                    ON CONFLICT (netkeiba_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        name_jp = EXCLUDED.name_jp,
                        sire_name = COALESCE(EXCLUDED.sire_name, horses.sire_name),
                        trainer_id = COALESCE(EXCLUDED.trainer_id, horses.trainer_id)
                    RETURNING id
                """),
                {
                    "name": horse.name,
                    "name_jp": horse.name_jp,
                    "sex": horse.sex,
                    "birth_year": horse.birth_year,
                    "netkeiba_id": horse.netkeiba_id,
                    "sire_name": horse.sire_name,
                    "trainer_id": trainer_db_id,
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
                        ON CONFLICT (name_jp) DO NOTHING
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
    """
    Scrape a single race by its netkeiba ID and save to DB.
    Tries result page first (for finished races), then shutuba (pre-race).
    """
    log.info(f"Scraping race {race_id}...")

    # Try result page first
    url = RESULT_URL.format(race_id=race_id)
    soup = _fetch(url)
    if soup:
        # Check if the result page actually has data (vs redirect to shutuba)
        has_results = soup.select_one("td.Result_Num") is not None
        if has_results:
            race = parse_result_page(soup, race_id)
            if race and race.entries:
                save_race_to_db(race)
                return race

    # Try shutuba page (pre-race entry list)
    url = SHUTUBA_URL.format(race_id=race_id)
    soup = _fetch(url)
    if soup:
        has_horses = bool(soup.select("tr.HorseList"))
        if has_horses:
            race = parse_shutuba_page(soup, race_id)
            if race and race.entries:
                save_race_to_db(race)
                return race

    log.warning(f"No entries parsed for race {race_id}")
    return None


def scrape_race_list(date_str: str) -> list[str]:
    """
    Scrape the race list for a given date and return netkeiba race IDs.
    Date format: YYYYMMDD

    The race list page loads race IDs via AJAX/JavaScript, so we also
    check the JS source for embedded race_id data.
    """
    # Netkeiba now loads races dynamically via race_list_sub.html
    url = BASE_URL + f"/top/race_list_sub.html?kaisai_date={date_str}"
    log.info(f"Fetching race list for {date_str}...")
    soup = _fetch(url)
    if not soup:
        return []

    race_ids = []

    # Method 1: Look for direct links (e.g. result.html?race_id=...)
    links = soup.find_all("a", href=re.compile(r"race_id=\d{12}"))
    for link in links:
        match = re.search(r"race_id=(\d{12})", link["href"])
        if match:
            rid = match.group(1)
            if rid not in race_ids:
                race_ids.append(rid)

    # Method 2: Look for race_id references in inline JS
    if not race_ids:
        page_text = str(soup)
        js_ids = re.findall(r'race_id["\']?\s*[:=]\s*["\']?(\d{12})', page_text)
        for rid in js_ids:
            if rid not in race_ids:
                race_ids.append(rid)

    # Method 3: Look for data attributes with race IDs
    if not race_ids:
        for el in soup.find_all(attrs={"data-race-id": True}):
            rid = el["data-race-id"]
            if re.match(r'^\d{12}$', rid) and rid not in race_ids:
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
