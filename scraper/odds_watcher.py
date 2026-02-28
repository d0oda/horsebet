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
import signal
import sys
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

    Uses netkeiba's JSON API (works during live betting window).
    Falls back to shutuba page HTML parsing for pre-race/static odds.
    """
    import json as _json

    # Method 1: JSON API (works during live betting window)
    api_url = (
        f"https://race.netkeiba.com/api/api_get_jra_odds.html"
        f"?race_id={race_id}&type=b1&action=update"
    )
    try:
        resp = requests.get(api_url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            data = _json.loads(resp.text)
            if data.get("status") not in ("NG",) and isinstance(data.get("data"), dict):
                odds_raw = data["data"].get("odds", [])
                if odds_raw and isinstance(odds_raw, list):
                    odds_list = []
                    for item in odds_raw:
                        # API format: each item has 'umaban' (horse number) and 'odds' (value)
                        horse_num = str(item.get("umaban", item.get("no", "")))
                        odds_val = item.get("odds", item.get("odds_value"))
                        if odds_val and horse_num:
                            try:
                                odds_list.append({
                                    "combination": horse_num,
                                    "odds_value": float(str(odds_val).replace(",", "")),
                                })
                            except (ValueError, TypeError):
                                continue
                    if odds_list:
                        log.debug(f"Got {len(odds_list)} odds from API for {race_id}")
                        return odds_list
    except Exception as e:
        log.debug(f"API odds fetch failed for {race_id}: {e}")

    # Method 2: Parse shutuba page (pre-race odds from HTML)
    shutuba_url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        resp = requests.get(shutuba_url, headers=HEADERS, timeout=30)
        resp.encoding = "EUC-JP"
        if resp.status_code != 200:
            log.warning(f"HTTP {resp.status_code} fetching shutuba for {race_id}")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        odds_list = []
        for row in soup.select("tr.HorseList"):
            cells = row.select("td")
            if len(cells) < 10:
                continue
            try:
                horse_num = cells[1].get_text(strip=True)
                odds_text = cells[9].get_text(strip=True)
                if not horse_num.isdigit():
                    continue
                if not odds_text or odds_text in ("---", "---.-", "**", "取消", "除外"):
                    continue
                odds_val = float(odds_text.replace(",", ""))
                odds_list.append({
                    "combination": horse_num,
                    "odds_value": odds_val,
                })
            except (ValueError, IndexError):
                continue

        if odds_list:
            log.debug(f"Got {len(odds_list)} odds from shutuba for {race_id}")
        return odds_list

    except requests.RequestException as e:
        log.warning(f"Error fetching shutuba for {race_id}: {e}")
        return []


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


def stream_odds(race_ids: list[str], interval_secs: int = 60):
    """
    Continuously stream odds for multiple races until interrupted (SIGINT).
    Polls each race at the given interval for better odds_slope / odds_late_money features.

    Args:
        race_ids: List of netkeiba race IDs (12-digit)
        interval_secs: Seconds between poll cycles (default 60)
    """
    running = True

    def _handle_sigint(sig, frame):
        nonlocal running
        log.info("\n⏹ Stopping odds streaming (SIGINT received)")
        running = False

    signal.signal(signal.SIGINT, _handle_sigint)

    log.info(f"📡 Streaming odds for {len(race_ids)} race(s) every {interval_secs}s")
    log.info(f"   Races: {', '.join(race_ids)}")
    log.info(f"   Press Ctrl+C to stop.")

    cycle = 0
    while running:
        cycle += 1
        for rid in race_ids:
            if not running:
                break
            odds = fetch_win_odds(rid)
            if odds:
                count = save_odds_snapshot(rid, odds)
                fav = min(o["odds_value"] for o in odds)
                log.info(f"  [cycle {cycle}] {rid}: {count} horses, fav={fav:.1f}x")
            else:
                log.warning(f"  [cycle {cycle}] {rid}: no odds data")

        if running:
            time.sleep(interval_secs)

    log.info(f"✅ Streaming stopped after {cycle} cycle(s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Odds Snapshot Watcher")
    parser.add_argument("--race", type=str, nargs="+", required=True, help="Race ID(s) to watch (12-digit)")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between snapshots (default 300)")
    parser.add_argument("--max", type=int, default=100, help="Max snapshots before stopping")
    parser.add_argument("--stream", action="store_true", help="Continuous streaming mode (runs until Ctrl+C)")

    args = parser.parse_args()

    if args.stream:
        stream_odds(args.race, interval_secs=args.interval)
    else:
        for race_id in args.race:
            watch_race(race_id, args.interval, args.max)


if __name__ == "__main__":
    main()
