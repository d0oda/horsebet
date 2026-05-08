#!/usr/bin/env python3
"""
UmaEdge — Live Race Watcher.

All-in-one race day script: fetches fresh odds, re-runs predictions,
and sends WhatsApp notifications with top 3 horses by EV before each race.

Usage:
    # Once mode — fetch odds, predict, notify upcoming races, then exit
    # (designed for GitHub Actions cron — runs every 10 min)
    python race_watcher.py --date 2026-05-09 --once

    # Continuous mode — loops all day
    python race_watcher.py --date 2026-05-09

    # Test mode: send all notifications immediately
    python race_watcher.py --date 2026-05-09 --test
"""

import argparse
import json
import logging
import time as time_module
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from scraper.db import get_session
from scraper.netkeiba import scrape_race
from notifications.dispatcher import notify_message

# Pipeline steps — same functions the main pipeline uses
from pipeline import step_odds, step_predict, step_frontend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("race_watcher")

JST = timezone(timedelta(hours=9))

# Default model config (same as pipeline)
MODEL_VERSION = "retrain_20260507_1645"
EV_THRESHOLD = 0.25
MAX_ODDS = 30.0
MIN_ODDS = 2.0

COURSE_NAMES = {
    1: 'Sapporo', 2: 'Hakodate', 3: 'Fukushima', 4: 'Niigata',
    5: 'Tokyo', 6: 'Nakayama', 7: 'Chukyo', 8: 'Kyoto',
    9: 'Hanshin', 10: 'Kokura',
}

STATE_DIR = Path("results")


def _state_path(date: str) -> Path:
    return STATE_DIR / f"watcher_state_{date}.json"


def load_state(date: str) -> dict:
    path = _state_path(date)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"notified": [], "venue_last_race": {}}


def save_state(date: str, state: dict):
    STATE_DIR.mkdir(exist_ok=True)
    with open(_state_path(date), "w") as f:
        json.dump(state, f, indent=2)


def refresh_pipeline(date: str) -> str:
    """
    Run the pipeline: fresh odds → predictions → data.json.
    Returns path to the predictions file.
    """
    log.info("━━━ Refreshing odds + predictions ━━━")

    # Get race netkeiba_ids from DB
    with get_session() as session:
        rows = session.execute(
            text("SELECT netkeiba_id FROM horsebet.races WHERE date = :d "
                 "ORDER BY course_id, race_number"),
            {"d": date},
        ).fetchall()
    race_ids = [r.netkeiba_id for r in rows]

    if not race_ids:
        log.error("No races in DB")
        return ""

    # 1. Fetch fresh odds
    step_odds(date, race_ids)

    # 2. Re-run predictions with fresh odds
    pred_path = step_predict(
        date, MODEL_VERSION, EV_THRESHOLD, MAX_ODDS, MIN_ODDS
    )

    # 3. Build data.json (dashboard + watcher source)
    step_frontend(pred_path, date)

    return pred_path


def load_data_json() -> list[dict]:
    """Load races from results/data.json."""
    path = Path("results/data.json")
    if not path.exists():
        log.error(f"No data.json at {path}")
        return []
    with open(path) as f:
        return json.load(f).get("races", [])


def load_race_schedule(date: str) -> list[dict]:
    """Load races from DB + data.json entries, sorted by post_time."""
    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT r.id, r.netkeiba_id, r.race_number, r.course_id,
                       r.post_time, r.distance, r.surface, r.race_name_jp
                FROM horsebet.races r
                WHERE r.date = :d
                ORDER BY r.post_time, r.course_id
            """),
            {"d": date},
        ).fetchall()

    data_races = load_data_json()
    if not data_races:
        return []

    # Index data.json by (venue_base, race_number)
    data_index = {}
    for dr in data_races:
        venue_base = dr.get("venue", "").split(" (")[0]
        key = (venue_base, dr.get("race_number"))
        data_index[key] = dr

    races = []
    for r in rows:
        if not r.post_time:
            continue
        race_date = datetime.strptime(date, "%Y-%m-%d").date()
        post_dt = datetime.combine(race_date, r.post_time, tzinfo=JST)
        venue = COURSE_NAMES.get(r.course_id, f"C{r.course_id}")

        data_race = data_index.get((venue, r.race_number))
        entries = data_race.get("entries", []) if data_race else []

        races.append({
            "id": r.id,
            "netkeiba_id": r.netkeiba_id,
            "race_number": r.race_number,
            "course_id": r.course_id,
            "venue": venue,
            "post_time": r.post_time.strftime("%H:%M"),
            "post_dt": post_dt,
            "distance": r.distance,
            "surface": (r.surface or "?").upper(),
            "race_name": r.race_name_jp or "",
            "entries": entries,
        })
    return races


def check_result(race: dict) -> dict | None:
    """Scrape result for a finished race."""
    try:
        race_data = scrape_race(race["netkeiba_id"])
        if not race_data or not race_data.entries:
            return None
        winner = None
        with get_session() as session:
            for entry in race_data.entries:
                if entry.finish_pos is not None:
                    session.execute(
                        text("""
                            UPDATE horsebet.entries
                            SET finish_pos = :fp, time_secs = :ts,
                                last_3f_secs = :l3f,
                                odds_win = COALESCE(:odds, odds_win)
                            WHERE race_id = :rid AND post_position = :pp
                        """),
                        {
                            "fp": entry.finish_pos, "ts": entry.time_secs,
                            "l3f": entry.last_3f_secs, "odds": entry.odds_win,
                            "rid": race["id"], "pp": entry.post_position,
                        },
                    )
                    if entry.finish_pos == 1:
                        winner = {
                            "horse_name": entry.horse_name,
                            "post_position": entry.post_position,
                            "odds": entry.odds_win or 0,
                            "time_secs": entry.time_secs,
                        }
            session.commit()
        return winner
    except Exception as e:
        log.warning(f"Could not fetch result: {e}")
        return None


def format_pre_race_message(race, top_entries, prev_result, prev_race, is_first):
    venue, rn, post = race["venue"], race["race_number"], race["post_time"]
    lines = []

    if not is_first and prev_race:
        pv, prn = prev_race["venue"], prev_race["race_number"]
        if prev_result:
            name = prev_result.get("horse_name", "?")
            pp = prev_result.get("post_position", "?")
            odds = prev_result.get("odds", 0)
            payout = int(odds * 1000) if odds else "?"
            lines.append(
                f"📋 *{pv} R{prn} Result*\n"
                f"   🏆 #{pp} {name}\n"
                f"   💰 ¥{payout:,} (odds {odds:.1f}x)\n"
            )
        else:
            lines.append(f"📋 *{pv} R{prn} Result*\n   ⏳ Not yet available\n")

    if lines:
        lines.append(f"{'─' * 24}\n")

    lines.append(
        f"🏇 *{venue} R{rn}* {post}\n"
        f"   {race['surface']} {race['distance']}m {race['race_name']}\n"
    )

    lines.append("📊 *Top 3 by EV:*")
    medals = ["🥇", "🥈", "🥉"]
    for i, e in enumerate(top_entries[:3]):
        draw_str = f"(枠{e['draw']})" if e.get("draw") else ""
        lines.append(
            f"{medals[i]} #{e['post_position']}{draw_str} *{e['horse_name']}*\n"
            f"    Odds: {e['odds']:.1f}x | EV: {e['ev']:+.1f}% | Win: {e['prob_combined']:.1f}%"
        )
    return "\n".join(lines)


def notify_race(race, prev_race, is_first_at_venue):
    venue, rn, post = race["venue"], race["race_number"], race["post_time"]
    log.info(f"📨 Preparing: {venue} R{rn} ({post})")

    prev_result = None
    if prev_race and not is_first_at_venue:
        log.info(f"   Checking result for {prev_race['venue']} R{prev_race['race_number']}...")
        time_module.sleep(1)
        prev_result = check_result(prev_race)
        if prev_result:
            log.info(f"   🏆 Winner: {prev_result['horse_name']} ({prev_result['odds']:.1f}x)")
        else:
            log.info("   ⏳ Result not yet available")

    entries = sorted(race.get("entries", []), key=lambda e: e.get("ev", -999), reverse=True)
    top3 = entries[:3]

    if not top3:
        log.info("   ⚠️ No entries for this race")
        return prev_result

    msg = format_pre_race_message(race, top3, prev_result, prev_race, is_first_at_venue)
    notify_message(msg)
    log.info(f"   ✅ Sent: {venue} R{rn} — Top EV: {top3[0]['horse_name']} ({top3[0]['ev']:+.1f}%)")
    return prev_result


def run_once(date: str, lead_time_min: int):
    """
    One-shot mode for GitHub Actions cron.
    Fetches fresh odds, re-predicts, notifies upcoming races, exits.
    """
    log.info(f"🏇 UmaEdge Watcher (once) — {date}")

    # 1. Refresh odds + predictions + data.json
    refresh_pipeline(date)

    # 2. Load fresh race schedule (with fresh data.json entries)
    races = load_race_schedule(date)
    if not races:
        log.error("No races found")
        return

    state = load_state(date)
    notified_ids = set(state.get("notified", []))
    venue_last_race_ids = state.get("venue_last_race", {})
    race_by_id = {r["id"]: r for r in races}

    now = datetime.now(JST)
    log.info(f"   JST: {now.strftime('%H:%M')} | "
             f"{len(races)} races | {len(notified_ids)} already notified")

    # 3. Find races ready to notify
    ready = []
    for race in races:
        if race["id"] in notified_ids:
            continue
        notify_at = race["post_dt"] - timedelta(minutes=lead_time_min)
        if now >= notify_at:
            ready.append(race)

    if not ready:
        remaining = [r for r in races if r["id"] not in notified_ids]
        if remaining:
            nxt = remaining[0]
            wait_min = (nxt["post_dt"] - timedelta(minutes=lead_time_min) - now).total_seconds() / 60
            log.info(f"   Next: {nxt['venue']} R{nxt['race_number']} at {nxt['post_time']} "
                     f"(in {wait_min:.0f} min)")
        else:
            log.info("   All races notified ✅")
        return

    log.info(f"   {len(ready)} race(s) ready to notify")

    # Startup message on first run
    if not notified_ids:
        venues = sorted(set(r["venue"] for r in races))
        startup_msg = (
            f"🏇 *UmaEdge — {date}*\n"
            f"Venues: {', '.join(venues)}\n"
            f"Races: {len(races)} ({races[0]['post_time']} – {races[-1]['post_time']})\n"
            f"Top 3 by EV before each race.\n"
            f"Let's go! 🍀"
        )
        notify_message(startup_msg)
        log.info("📨 Startup message sent")
        time_module.sleep(2)

    # 4. Notify
    for race in ready:
        venue = race["venue"]
        prev_race_id = venue_last_race_ids.get(venue)
        prev_race = race_by_id.get(prev_race_id) if prev_race_id else None

        notify_race(race, prev_race, prev_race is None)

        notified_ids.add(race["id"])
        venue_last_race_ids[venue] = race["id"]
        time_module.sleep(3)

    if len(notified_ids) >= len(races):
        notify_message("🏁 All races complete. See you next time! 🏇")

    state["notified"] = list(notified_ids)
    state["venue_last_race"] = venue_last_race_ids
    save_state(date, state)
    log.info(f"   State saved ({len(notified_ids)}/{len(races)} notified)")


def run_continuous(date: str, lead_time_min: int, test_mode: bool):
    """Continuous mode — refreshes odds + re-predicts before each notification."""
    log.info(f"🏇 UmaEdge Race Watcher — {date}")

    # Initial refresh
    refresh_pipeline(date)
    races = load_race_schedule(date)
    if not races:
        log.error("No races found")
        return

    log.info(f"   {len(races)} races across "
             f"{len(set(r['venue'] for r in races))} venues")

    venue_last_race: dict[str, dict] = {}
    notified = set()

    venues = sorted(set(r["venue"] for r in races))
    startup_msg = (
        f"🏇 *UmaEdge Watcher — {date}*\n"
        f"Venues: {', '.join(venues)}\n"
        f"Races: {len(races)} ({races[0]['post_time']} – {races[-1]['post_time']})\n"
        f"Sending top 3 by EV before each race.\n"
        f"Let's go! 🍀"
    )
    notify_message(startup_msg)
    log.info("📨 Startup message sent")

    while True:
        now = datetime.now(JST)

        next_race = None
        for race in races:
            if race["id"] in notified:
                continue
            notify_at = race["post_dt"] - timedelta(minutes=lead_time_min)
            if test_mode or now >= notify_at:
                next_race = race
                break

        if next_race is None:
            if len(notified) >= len(races):
                log.info("✅ All races notified.")
                notify_message("🏁 All races complete. See you next time! 🏇")
                break
            remaining = [r for r in races if r["id"] not in notified]
            if remaining:
                wait_until = remaining[0]["post_dt"] - timedelta(minutes=lead_time_min)
                wait_secs = (wait_until - now).total_seconds()
                if wait_secs > 0:
                    log.info(f"⏰ Next: {remaining[0]['venue']} R{remaining[0]['race_number']} "
                             f"at {remaining[0]['post_time']} (in {wait_secs/60:.0f} min)")
                    time_module.sleep(min(wait_secs, 60))
                continue
            break

        # Refresh odds + predictions before each notification
        if not test_mode:
            refresh_pipeline(date)
            races = load_race_schedule(date)
            # Re-find the race by id
            race_by_id = {r["id"]: r for r in races}
            next_race = race_by_id.get(next_race["id"], next_race)

        race = next_race
        venue = race["venue"]
        prev_race = venue_last_race.get(venue)

        notify_race(race, prev_race, prev_race is None)

        notified.add(race["id"])
        venue_last_race[venue] = race

        time_module.sleep(3 if test_mode else 5)


def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Live Race Watcher")
    parser.add_argument("--date", required=True, help="Race date (YYYY-MM-DD)")
    parser.add_argument("--lead-time", type=int, default=5,
                        help="Minutes before post time to send (default: 5)")
    parser.add_argument("--once", action="store_true",
                        help="One-shot mode: refresh, notify, exit (for cron)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: notify all races immediately")
    args = parser.parse_args()

    try:
        if args.once:
            run_once(args.date, args.lead_time)
        else:
            run_continuous(args.date, args.lead_time, args.test)
    except KeyboardInterrupt:
        log.info("\n👋 Watcher stopped by user")


if __name__ == "__main__":
    main()
