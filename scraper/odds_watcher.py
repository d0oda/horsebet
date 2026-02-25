"""
UmaEdge — Odds Snapshot Watcher.

Periodically captures win/place odds from netkeiba for upcoming races.
Stores snapshots as time-series in the horsebet.odds_snapshots table.

Usage:
    # Watch odds for a specific race (snapshot every 5 min)
    python -m scraper.odds_watcher --race 202506010101

    # Watch all races for today
    python -m scraper.odds_watcher --date 20250628 --interval 300
"""

import argparse
import logging
import os
import re
import time
from datetime import datetime
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
log = logging.getLogger("odds_watcher")

ODDS_URL = "https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b1"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}


def fetch_win_odds(race_id: str) -> list[dict]:
    """
    Fetch current win odds for all horses in a race.
    Returns list of dicts: [{combination: "1", odds_value: 3.5}, ...]
    """
    url = ODDS_URL.format(race_id=race_id)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.encoding = "utf-8"
        if resp.status_code != 200:
            log.warning(f"HTTP {resp.status_code} fetching odds for {race_id}")
            return []
    except requests.RequestException as e:
        log.warning(f"Error fetching odds for {race_id}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    odds_list = []

    # Parse the odds table
    odds_table = soup.find("table", class_="RaceOdds_HorseList_Table")
    if not odds_table:
        # Fallback: try to find odds in different format
        odds_divs = soup.find_all("tr", class_=re.compile(r"HorseList"))
        for div in odds_divs:
            cells = div.find_all("td")
            if len(cells) >= 4:
                try:
                    horse_num = cells[1].get_text(strip=True)
                    odds_text = cells[3].get_text(strip=True)
                    odds_val = float(odds_text) if odds_text and odds_text != "---" else None
                    if odds_val and horse_num.isdigit():
                        odds_list.append({
                            "combination": horse_num,
                            "odds_value": odds_val,
                        })
                except (ValueError, IndexError):
                    continue
    else:
        rows = odds_table.find_all("tr")[1:]
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 4:
                try:
                    horse_num = cells[1].get_text(strip=True)
                    odds_text = cells[3].get_text(strip=True)
                    odds_val = float(odds_text) if odds_text and odds_text != "---" else None
                    if odds_val and horse_num.isdigit():
                        odds_list.append({
                            "combination": horse_num,
                            "odds_value": odds_val,
                        })
                except (ValueError, IndexError):
                    continue

    return odds_list


def save_odds_snapshot(race_id: str, odds: list[dict]) -> int:
    """Save a batch of odds snapshots to the database. Returns count inserted."""
    if not odds:
        return 0

    now = datetime.utcnow()

    with get_session() as session:
        # Look up internal race_id
        race_db = session.execute(
            text("SELECT id FROM races WHERE netkeiba_id = :nid"),
            {"nid": race_id},
        ).fetchone()

        if not race_db:
            log.warning(f"Race {race_id} not found in DB — scrape the race first")
            return 0

        race_db_id = race_db[0]
        count = 0

        for odd in odds:
            session.execute(
                text("""
                    INSERT INTO odds_snapshots (race_id, captured_at, bet_type, combination, odds_value)
                    VALUES (:race_id, :captured_at, 'win', :combination, :odds_value)
                """),
                {
                    "race_id": race_db_id,
                    "captured_at": now,
                    "combination": odd["combination"],
                    "odds_value": odd["odds_value"],
                },
            )
            count += 1

        log.info(f"💾 Saved {count} odds snapshots for race {race_id} at {now.strftime('%H:%M:%S')}")
        return count


def watch_race(race_id: str, interval_secs: int = 300, max_snapshots: int = 100):
    """
    Continuously poll odds for a race at the given interval.

    Args:
        race_id: Netkeiba race ID (12-digit)
        interval_secs: Seconds between snapshots (default 300 = 5 min)
        max_snapshots: Maximum number of snapshots before stopping
    """
    log.info(f"👁 Watching odds for race {race_id} every {interval_secs}s (max {max_snapshots} snapshots)")

    for i in range(max_snapshots):
        odds = fetch_win_odds(race_id)
        if odds:
            save_odds_snapshot(race_id, odds)
            log.info(f"  Snapshot {i + 1}/{max_snapshots}: {len(odds)} horses, "
                     f"fav={min(o['odds_value'] for o in odds):.1f}x")
        else:
            log.warning(f"  Snapshot {i + 1}: no odds data")

        if i < max_snapshots - 1:
            time.sleep(interval_secs)

    log.info(f"✅ Finished watching race {race_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Odds Snapshot Watcher")
    parser.add_argument("--race", type=str, required=True, help="Race ID to watch (12-digit)")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between snapshots (default 300)")
    parser.add_argument("--max", type=int, default=100, help="Max snapshots before stopping")

    args = parser.parse_args()
    watch_race(args.race, args.interval, args.max)


if __name__ == "__main__":
    main()
