"""
UmaEdge — Reverse-Engineered JRA Odds Watcher & API Client.

Fetches real-time win, place, and exotic odds directly from Netkeiba's internal
JRA odds JSON API (https://race.netkeiba.com/api/api_get_jra_odds.html).

Supported Bet Types:
    - Win (単勝) & Place (複勝) -> type=1
    - Bracket Quinella (枠連)   -> type=3
    - Quinella (馬連)          -> type=4
    - Wide (ワイド)             -> type=5
    - Exacta (馬単)            -> type=6
    - Trio (3連複)              -> type=7
    - Trifecta (3連単)          -> type=8

Usage:
    # 1. Fetch & print live win/place odds for a race:
    python -m scraper.odds_watcher --race 202405021211

    # 2. Fetch full exotic pools (exacta, trio, trifecta):
    python -m scraper.odds_watcher --race 202405021211 --exotics

    # 3. Stream live odds every 30s before post time:
    python -m scraper.odds_watcher --race 202405021211 --stream --interval 30

    # 4. Save snapshots to database:
    python -m scraper.odds_watcher --race 202405021211 --save
"""

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
from datetime import datetime
from typing import Any, Optional

import requests
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

API_BASE_URL = "https://race.netkeiba.com/api/api_get_jra_odds.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "X-Requested-With": "XMLHttpRequest",
}

# API Type -> Internal Bet Pool Identifier
TYPE_TO_BET_TYPE = {
    "1": ("win", "place"),
    "3": ("bracket_quinella",),
    "4": ("quinella",),
    "5": ("wide",),
    "6": ("exacta",),
    "7": ("trio",),
    "8": ("trifecta",),
}

BET_TYPE_TO_TYPE = {
    "win": "1",
    "place": "1",
    "bracket_quinella": "3",
    "wakuren": "3",
    "quinella": "4",
    "umaren": "4",
    "wide": "5",
    "exacta": "6",
    "umatan": "6",
    "trio": "7",
    "sanrenpuku": "7",
    "trifecta": "8",
    "sanrentan": "8",
}


def _clean_odds_val(val_str: Any) -> Optional[float]:
    """Parse raw odds string into float, handling commas and invalid markers."""
    if val_str is None:
        return None
    s = str(val_str).strip().replace(",", "")
    if s in ("", "---", "取消", "除外", "0", "0.0", "0.00", "null", "None"):
        return None
    try:
        val = float(s)
        return val if val > 0.0 else None
    except (ValueError, TypeError):
        return None


def _format_combination(raw_key: str) -> str:
    """
    Format raw horse key into dash-separated combination string.
    - '01' -> '1'
    - '0102' -> '1-2'
    - '010203' -> '1-2-3'
    """
    raw_key = str(raw_key).strip()
    if len(raw_key) == 2:
        return str(int(raw_key))
    
    chunks = [str(int(raw_key[i:i+2])) for i in range(0, len(raw_key), 2)]
    return "-".join(chunks)


def _request_odds_api(race_id: str, api_type: str, max_retries: int = 3) -> Optional[dict]:
    """Make HTTP GET request to Netkeiba internal odds API with retry logic."""
    url = f"{API_BASE_URL}?race_id={race_id}&type={api_type}&action=init"
    headers = dict(HEADERS)
    headers["Referer"] = f"https://race.netkeiba.com/odds/index.html?race_id={race_id}"

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=(4.0, 8.0))
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") != "NG" and isinstance(data.get("data"), dict):
                    return data["data"]
            log.warning(
                f"Odds API fetch failed for {race_id} (type={api_type}) "
                f"attempt {attempt + 1}/{max_retries}: status {resp.status_code}"
            )
        except Exception as e:
            log.warning(
                f"Odds API exception for {race_id} (type={api_type}) "
                f"attempt {attempt + 1}/{max_retries}: {e}"
            )
        if attempt < max_retries - 1:
            time.sleep(1.0 * (attempt + 1))
    return None


def fetch_win_odds(race_id: str) -> list[dict]:
    """
    Fetch current win odds for all horses in a race.
    Returns list of dicts:
        [{'combination': '1', 'odds_value': 3.5, 'popularity': 2}, ...]

    100% backwards-compatible with legacy UmaEdge pipeline and watchers.
    """
    odds_data = _request_odds_api(race_id, api_type="1")
    if not odds_data:
        return []

    tansho = odds_data.get("odds", {}).get("1", {})
    if not isinstance(tansho, dict):
        return []

    odds_list = []
    for horse_num, values in tansho.items():
        if isinstance(values, list) and len(values) >= 1:
            odds_val = _clean_odds_val(values[0])
            if odds_val is None:
                continue

            pop = int(values[2]) if len(values) >= 3 and str(values[2]).isdigit() else None
            odds_list.append({
                "combination": str(int(horse_num)),
                "odds_value": odds_val,
                "popularity": pop,
                "bet_type": "win",
            })

    if odds_list:
        log.debug(f"Got {len(odds_list)} win odds for {race_id}")
    return odds_list


def fetch_place_odds(race_id: str) -> list[dict]:
    """
    Fetch current place (複勝) odds for all horses in a race.
    Returns list of dicts:
        [{'combination': '1', 'min_odds': 1.5, 'max_odds': 2.2, 'popularity': 2}, ...]
    """
    odds_data = _request_odds_api(race_id, api_type="1")
    if not odds_data:
        return []

    fukusho = odds_data.get("odds", {}).get("2", {})
    if not isinstance(fukusho, dict):
        return []

    place_list = []
    for horse_num, values in fukusho.items():
        if isinstance(values, list) and len(values) >= 2:
            min_odds = _clean_odds_val(values[0])
            max_odds = _clean_odds_val(values[1])
            if min_odds is None:
                continue

            pop = int(values[2]) if len(values) >= 3 and str(values[2]).isdigit() else None
            place_list.append({
                "combination": str(int(horse_num)),
                "min_odds": min_odds,
                "max_odds": max_odds if max_odds is not None else min_odds,
                "popularity": pop,
                "bet_type": "place",
            })

    return place_list


def fetch_exotic_odds(race_id: str, bet_types: Optional[list[str]] = None) -> list[dict]:
    """
    Fetch current exotic odds for a race.
    Supported pools: 'quinella', 'wide', 'exacta', 'trio', 'trifecta', 'bracket_quinella'.

    Returns list of dicts:
        [{'bet_type': 'exacta', 'combination': '1-2', 'odds_value': 35.2, 'popularity': 12}, ...]
    """
    api_types = {
        "4": "quinella",
        "5": "wide",
        "6": "exacta",
        "7": "trio",
        "8": "trifecta",
    }
    if bet_types:
        filter_set = {b.lower() for b in bet_types}
        api_types = {k: v for k, v in api_types.items() if v in filter_set or BET_TYPE_TO_TYPE.get(v) in filter_set}

    odds_list = []
    for api_type, bet_type in api_types.items():
        data = _request_odds_api(race_id, api_type=api_type)
        if not data:
            continue

        pool = data.get("odds", {}).get(api_type, {})
        if not isinstance(pool, dict):
            continue

        for horse_key, values in pool.items():
            if isinstance(values, list) and len(values) >= 1:
                odds_val = _clean_odds_val(values[0])
                if odds_val is None:
                    continue

                combination = _format_combination(horse_key)
                pop = int(values[2]) if len(values) >= 3 and str(values[2]).isdigit() else None

                entry = {
                    "bet_type": bet_type,
                    "combination": combination,
                    "odds_value": odds_val,
                    "popularity": pop,
                }
                if bet_type == "wide" and len(values) >= 2:
                    max_odds = _clean_odds_val(values[1])
                    entry["max_odds"] = max_odds if max_odds is not None else odds_val

                odds_list.append(entry)

        # Politeness delay between exotic pools
        time.sleep(0.3)

    if odds_list:
        log.info(f"Got {len(odds_list)} exotic odds from API for {race_id}")
    return odds_list


def fetch_race_odds(race_id: str, include_exotics: bool = False) -> dict[str, Any]:
    """
    Fetch comprehensive odds package for a single race.
    Always includes Win and Place in 1 network request.
    Optionally fetches exotic pools (quinella, wide, exacta, trio, trifecta).
    """
    win_odds = fetch_win_odds(race_id)
    place_odds = fetch_place_odds(race_id)

    result = {
        "race_id": race_id,
        "win": win_odds,
        "place": place_odds,
        "captured_at": datetime.utcnow().isoformat(),
    }

    if include_exotics:
        exotics = fetch_exotic_odds(race_id)
        grouped_exotics = {}
        for item in exotics:
            btype = item["bet_type"]
            grouped_exotics.setdefault(btype, []).append(item)
        result.update(grouped_exotics)

    return result


def update_race_odds_in_db(race_netkeiba_id: str, odds: Optional[list[dict]] = None) -> int:
    """
    Update `entries.odds_win` and `entries.popularity` in the DB for a race.
    If `odds` is None, automatically fetches fresh win odds via `fetch_win_odds`.
    
    Returns count of updated entries.
    """
    if odds is None:
        odds = fetch_win_odds(race_netkeiba_id)
    if not odds:
        return 0

    with get_session() as session:
        race_row = session.execute(
            text("SELECT id FROM races WHERE netkeiba_id = :nid"),
            {"nid": race_netkeiba_id},
        ).fetchone()

        if not race_row:
            log.warning(f"Race {race_netkeiba_id} not found in DB")
            return 0

        db_race_id = race_row[0]
        updated_count = 0

        for o in odds:
            if o.get("bet_type", "win") != "win":
                continue
            try:
                pp = int(o["combination"])
                odds_val = o["odds_value"]
                pop = o.get("popularity")

                result = session.execute(
                    text("""
                        UPDATE entries
                        SET odds_win = :odds,
                            popularity = COALESCE(:pop, popularity)
                        WHERE race_id = :race_id AND post_position = :pp
                    """),
                    {"odds": odds_val, "pop": pop, "race_id": db_race_id, "pp": pp},
                )
                updated_count += result.rowcount
            except Exception as e:
                log.debug(f"Failed updating entry {o}: {e}")

        session.commit()
        return updated_count


def save_odds_snapshot(race_id: str, odds: list[dict]) -> int:
    """
    Save a batch of odds snapshots to the database for odds drift / slope tracking.
    Returns count inserted.
    """
    if not odds:
        return 0

    now = datetime.utcnow()

    with get_session() as session:
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
            odds_val = odd.get("odds_value") or odd.get("min_odds")
            if odds_val is None:
                continue

            session.execute(
                text("""
                    INSERT INTO odds_snapshots (race_id, captured_at, bet_type, combination, odds_value)
                    VALUES (:race_id, :captured_at, :bet_type, :combination, :odds_value)
                """),
                {
                    "race_id": race_db_id,
                    "captured_at": now,
                    "bet_type": odd.get("bet_type", "win"),
                    "combination": odd["combination"],
                    "odds_value": float(odds_val),
                },
            )
            count += 1

        session.commit()
        log.info(f"💾 Saved {count} odds snapshots for race {race_id} at {now.strftime('%H:%M:%S')}")
        return count


def stream_odds(race_ids: list[str], interval_secs: int = 60, save_to_db: bool = True):
    """
    Continuously stream odds for multiple races until interrupted (Ctrl+C).
    Polls each race at the given interval for drift / late money tracking.
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
            win_odds = fetch_win_odds(rid)
            place_odds = fetch_place_odds(rid)
            all_odds = win_odds + place_odds

            if win_odds:
                fav = min(o["odds_value"] for o in win_odds)
                fav_horse = [o["combination"] for o in win_odds if o["odds_value"] == fav][0]
                log.info(f"  [cycle {cycle}] {rid}: {len(win_odds)} horses | Fav #{fav_horse} @ {fav:.1f}x")
                if save_to_db:
                    update_race_odds_in_db(rid, win_odds)
                    save_odds_snapshot(rid, all_odds)
            else:
                log.warning(f"  [cycle {cycle}] {rid}: no odds data returned")

        if running:
            time.sleep(interval_secs)

    log.info(f"✅ Streaming stopped after {cycle} cycle(s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Live JRA Odds Watcher & Client")
    parser.add_argument("--race", type=str, nargs="+", required=True, help="Race ID(s) (12-digit netkeiba ID)")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds for streaming (default 60)")
    parser.add_argument("--exotics", action="store_true", help="Also fetch exotic odds pools (Quinella, Trio, Trifecta)")
    parser.add_argument("--save", action="store_true", help="Save live odds into entries and odds_snapshots DB")
    parser.add_argument("--stream", action="store_true", help="Continuous polling mode until Ctrl+C")

    args = parser.parse_args()

    if args.stream:
        stream_odds(args.race, interval_secs=args.interval, save_to_db=args.save)
    else:
        for race_id in args.race:
            print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"📊 LIVE ODDS FOR RACE: {race_id}")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            odds_pkg = fetch_race_odds(race_id, include_exotics=args.exotics)
            
            win_map = {o["combination"]: o for o in odds_pkg.get("win", [])}
            place_map = {o["combination"]: o for o in odds_pkg.get("place", [])}

            print(f"{'#':<4} {'Win Odds':<12} {'Place Odds Range':<20} {'Pop Rank':<10}")
            print("-" * 50)
            for comb in sorted(win_map.keys(), key=lambda x: int(x)):
                w = win_map.get(comb, {})
                p = place_map.get(comb, {})
                w_str = f"{w.get('odds_value', '-'):.1f}x" if isinstance(w.get('odds_value'), (int, float)) else "-"
                p_min = p.get('min_odds')
                p_max = p.get('max_odds')
                p_str = f"{p_min:.1f} - {p_max:.1f}x" if p_min is not None and p_max is not None else "-"
                pop_str = f"#{w.get('popularity', '-')}"
                print(f"{comb:<4} {w_str:<12} {p_str:<20} {pop_str:<10}")

            if args.exotics:
                for pool in ["quinella", "exacta", "wide", "trio", "trifecta"]:
                    items = odds_pkg.get(pool, [])
                    if items:
                        print(f"\n🏷 {pool.upper()} (Top 5):")
                        top_items = sorted(items, key=lambda x: x["odds_value"])[:5]
                        for it in top_items:
                            print(f"  Combo: {it['combination']:<10} Odds: {it['odds_value']:<8.1f} (Rank #{it.get('popularity', '-')})")

            if args.save:
                updated = update_race_odds_in_db(race_id, odds_pkg.get("win", []))
                saved = save_odds_snapshot(race_id, odds_pkg.get("win", []) + odds_pkg.get("place", []))
                print(f"\n💾 Updated DB: {updated} entries updated, {saved} snapshot rows created.")


if __name__ == "__main__":
    main()
