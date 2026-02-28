#!/usr/bin/env python3
"""
UmaEdge — Race Day Pipeline.

Single command to scrape, predict, and build frontend data for a race day.

Usage:
    # Full pipeline
    python pipeline.py --date 2026-03-01

    # Skip scraping (races already in DB)
    python pipeline.py --date 2026-03-01 --skip-scrape

    # Skip odds fetch (e.g. race day hasn't started yet)
    python pipeline.py --date 2026-03-01 --skip-odds

    # Custom model / thresholds
    python pipeline.py --date 2026-03-01 --version 2026_v2 --ev-threshold 0.08
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from sqlalchemy import text

from scraper.db import get_session
from scraper.netkeiba import scrape_race_list, scrape_race, save_race_to_db
from scraper.odds_watcher import fetch_win_odds
from models.predict_final import predict_with_filters
from results.build_data_json import build_data_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# Odds fetch window: how far ahead (min) to look for upcoming races
ODDS_WINDOW_MIN = 20   # fetch for races starting within 20 min
ODDS_FINISHED_GRACE = 5  # skip races finished more than 5 min ago


def step_scrape(date: str) -> list[str]:
    """Step 1: Discover and scrape all races for the date."""
    date_compact = date.replace("-", "")
    log.info(f"━━━ Step 1: Scraping races for {date} ━━━")

    race_ids = scrape_race_list(date_compact)
    if not race_ids:
        log.error(f"No races found for {date}. Is the date correct?")
        return []

    log.info(f"Found {len(race_ids)} race IDs")

    # Check which are already in DB
    with get_session() as session:
        existing = session.execute(
            text("SELECT netkeiba_id FROM horsebet.races WHERE date = :d"),
            {"d": date},
        ).fetchall()
    existing_ids = {r.netkeiba_id for r in existing}

    new_ids = [rid for rid in race_ids if rid not in existing_ids]
    if not new_ids:
        log.info(f"All {len(race_ids)} races already in DB — skipping scrape")
    else:
        log.info(f"Scraping {len(new_ids)} new races ({len(existing_ids)} already in DB)...")
        for i, rid in enumerate(new_ids, 1):
            log.info(f"  [{i}/{len(new_ids)}] {rid}")
            try:
                race_data = scrape_race(rid)
                if race_data:
                    save_race_to_db(race_data)
            except Exception as e:
                log.warning(f"  ⚠️ Failed: {e}")
            time.sleep(1)  # rate limit

    return race_ids


def step_odds(date: str, race_ids: list[str]) -> list[int]:
    """
    Step 2: Fetch current odds — only for upcoming races (smart scheduling).

    Returns list of DB race IDs that were updated (for targeted re-prediction).
    """
    JST = timezone(timedelta(hours=9))
    now_jst = datetime.now(JST)
    log.info(f"━━━ Step 2: Smart odds fetch (JST: {now_jst.strftime('%H:%M')}) ━━━")

    # Get race info including post_time
    with get_session() as session:
        races = session.execute(
            text("""
                SELECT id, netkeiba_id, race_number, post_time
                FROM horsebet.races WHERE date = :d
                ORDER BY course_id, race_number
            """),
            {"d": date},
        ).fetchall()
    nk_to_db = {r.netkeiba_id: r.id for r in races}
    nk_to_info = {r.netkeiba_id: r for r in races}

    # Filter: only races starting within ODDS_WINDOW_MIN
    upcoming_ids = []
    skipped_past = 0
    skipped_future = 0
    no_time = 0
    for nk_id in race_ids:
        info = nk_to_info.get(nk_id)
        if not info or not info.post_time:
            # No post_time data — include it (fallback)
            upcoming_ids.append(nk_id)
            no_time += 1
            continue

        # post_time is a time object from DB
        post_dt = datetime.combine(now_jst.date(), info.post_time, tzinfo=JST)
        mins_until = (post_dt - now_jst).total_seconds() / 60

        if mins_until < -ODDS_FINISHED_GRACE:
            # Race finished — skip
            skipped_past += 1
        elif mins_until > ODDS_WINDOW_MIN:
            # Race too far away — next run will catch it
            skipped_future += 1
        else:
            # Race is upcoming — fetch odds
            upcoming_ids.append(nk_id)

    if not upcoming_ids:
        log.info(f"  No upcoming races (past: {skipped_past}, future: {skipped_future})")
        return []

    log.info(f"  Targeting {len(upcoming_ids)} races "
             f"(past: {skipped_past}, future: {skipped_future}, no_time: {no_time})")

    updated_db_ids = []
    for i, nk_id in enumerate(upcoming_ids, 1):
        db_id = nk_to_db.get(nk_id)
        if not db_id:
            continue

        info = nk_to_info.get(nk_id)
        time_str = info.post_time.strftime('%H:%M') if info and info.post_time else "??:??"

        odds = fetch_win_odds(nk_id)
        if not odds:
            log.warning(f"  [{i}/{len(upcoming_ids)}] R{info.race_number if info else '?'} ({time_str}): no odds")
            time.sleep(0.5)
            continue

        with get_session() as session:
            count = 0
            for o in odds:
                pp = int(o["combination"])
                result = session.execute(
                    text("""
                        UPDATE horsebet.entries
                        SET odds_win = :odds
                        WHERE race_id = :race_id AND post_position = :pp
                    """),
                    {"odds": o["odds_value"], "race_id": db_id, "pp": pp},
                )
                count += result.rowcount
            session.commit()

        log.info(f"  [{i}/{len(upcoming_ids)}] R{info.race_number if info else '?'} "
                 f"({time_str}): {len(odds)} horses updated")
        updated_db_ids.append(db_id)
        time.sleep(0.5)

    log.info(f"✅ Updated odds for {len(updated_db_ids)} races")
    return updated_db_ids


def step_predict(date: str, version: str, ev_threshold: float,
                 max_odds: float, min_odds: float,
                 race_db_ids: list[int] = None) -> str:
    """
    Step 3: Run model predictions. If race_db_ids is provided, only re-predict
    those races and merge into existing predictions file. Otherwise predict all.
    """
    output_path = f"results/predictions_{date}.json"

    # Get all race IDs from DB
    with get_session() as session:
        rows = session.execute(
            text("SELECT id FROM horsebet.races WHERE date = :d ORDER BY course_id, race_number"),
            {"d": date},
        ).fetchall()
    all_race_ids = [r.id for r in rows]

    if not all_race_ids:
        log.error(f"No races in DB for {date}")
        sys.exit(1)

    # Decide which races to predict
    if race_db_ids:
        # Incremental: only re-predict the updated races
        predict_ids = race_db_ids
        log.info(f"━━━ Step 3: Re-predicting {len(predict_ids)} updated races ({version}) ━━━")
    else:
        # Full: predict everything
        predict_ids = all_race_ids
        log.info(f"━━━ Step 3: Predicting all {len(predict_ids)} races ({version}) ━━━")

    df = predict_with_filters(
        race_ids=predict_ids,
        model_version=version,
        ev_threshold=ev_threshold,
        max_odds=max_odds,
        min_odds=min_odds,
    )

    if df.empty:
        log.error("No predictions generated")
        if not race_db_ids:
            sys.exit(1)
        return output_path

    # If incremental, merge with existing predictions
    if race_db_ids and Path(output_path).exists():
        with open(output_path) as f:
            existing = json.load(f)

        # Replace predictions for updated races, keep the rest
        updated_race_ids = set(race_db_ids)
        kept = [p for p in existing.get("predictions", [])
                if p.get("race_id") not in updated_race_ids]
        new_preds = df.to_dict(orient="records")
        all_preds = kept + new_preds

        existing["predictions"] = all_preds
        existing["summary"]["total_entries"] = len(all_preds)
        existing["summary"]["value_bets"] = sum(1 for p in all_preds if p.get("is_value_bet"))

        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        log.info(f"✅ Merged {len(new_preds)} updated + {len(kept)} existing predictions")
    else:
        # Save full predictions
        value_bets = df[df["is_value_bet"]]
        output_data = {
            "model": version,
            "date": date,
            "filters": {
                "ev_threshold": ev_threshold,
                "max_odds": max_odds,
                "min_odds": min_odds,
            },
            "summary": {
                "total_races": len(all_race_ids),
                "total_entries": len(df),
                "value_bets": len(value_bets),
            },
            "predictions": df.to_dict(orient="records"),
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        log.info(f"✅ {len(predict_ids)} races, {len(df)} entries, {len(value_bets)} value bets")

    log.info(f"   Saved to {output_path}")
    return output_path


def step_frontend(predictions_path: str, date: str):
    """Step 4: Build data.json and update index.html."""
    log.info("━━━ Step 4: Building frontend ━━━")

    build_data_json(predictions_path, date=date, output="results/data.json")

    # Update index.html date
    html_path = Path("results/index.html")
    if html_path.exists():
        content = html_path.read_text()

        # Parse date for display
        dt = datetime.strptime(date, "%Y-%m-%d")
        weekday_jp = ["月", "火", "水", "木", "金", "土", "日"][dt.weekday()]
        date_jp = f"{dt.year}年{dt.month}月{dt.day}日 ({weekday_jp})"

        # Replace date in title and display
        content = re.sub(
            r'<span id="race-date">.*?</span>',
            f'<span id="race-date">{date_jp}</span>',
            content,
        )
        content = re.sub(
            r'(UmaEdge\s*—\s*)\w+ \d+, \d+',
            f'\\g<1>{dt.strftime("%b %d, %Y")}',
            content,
        )
        html_path.write_text(content)
        log.info(f"✅ Updated index.html → {date_jp}")

    log.info("━━━ Pipeline complete ━━━")


def main():
    parser = argparse.ArgumentParser(
        description="UmaEdge — Race Day Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --date 2026-03-01
  python pipeline.py --date 2026-03-01 --skip-scrape --skip-odds
  python pipeline.py --date 2026-03-01 --version 2026_v2 --ev-threshold 0.08
        """,
    )
    parser.add_argument("--date", required=True, help="Race date (YYYY-MM-DD)")
    parser.add_argument("--version", default="2026_v2", help="Model version (default: 2026_v2)")
    parser.add_argument("--ev-threshold", type=float, default=0.05, help="Min EV for value bet (default: 5%%)")
    parser.add_argument("--max-odds", type=float, default=30.0, help="Max odds filter (default: 30)")
    parser.add_argument("--min-odds", type=float, default=1.5, help="Min odds filter (default: 1.5)")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip race scraping (already in DB)")
    parser.add_argument("--skip-odds", action="store_true", help="Skip odds fetching")
    parser.add_argument("--skip-predict", action="store_true", help="Skip prediction (reuse existing)")
    args = parser.parse_args()

    log.info(f"🏇 UmaEdge Pipeline — {args.date}")
    log.info(f"   Model: {args.version} | EV threshold: {args.ev_threshold:.0%} | "
             f"Odds: {args.min_odds}-{args.max_odds}x")

    # Step 1: Scrape
    if args.skip_scrape:
        log.info("━━━ Step 1: Scraping → SKIPPED ━━━")
        with get_session() as session:
            rows = session.execute(
                text("SELECT netkeiba_id FROM horsebet.races WHERE date = :d ORDER BY course_id, race_number"),
                {"d": args.date},
            ).fetchall()
        race_ids = [r.netkeiba_id for r in rows]
        log.info(f"   Found {len(race_ids)} races in DB")
    else:
        race_ids = step_scrape(args.date)

    if not race_ids:
        log.error("No races found — aborting")
        sys.exit(1)

    # Step 2: Odds (returns list of updated DB race IDs)
    updated_db_ids = []
    if args.skip_odds:
        log.info("━━━ Step 2: Odds → SKIPPED ━━━")
    else:
        updated_db_ids = step_odds(args.date, race_ids)

    # Step 3: Predict
    pred_path = f"results/predictions_{args.date}.json"
    if args.skip_predict and Path(pred_path).exists():
        log.info(f"━━━ Step 3: Predict → SKIPPED (using {pred_path}) ━━━")
    elif updated_db_ids:
        # Incremental: only re-predict races with freshly updated odds
        pred_path = step_predict(args.date, args.version, args.ev_threshold,
                                 args.max_odds, args.min_odds,
                                 race_db_ids=updated_db_ids)
    elif not Path(pred_path).exists():
        # First run: predict all
        pred_path = step_predict(args.date, args.version, args.ev_threshold,
                                 args.max_odds, args.min_odds)
    else:
        # No odds updated and predictions exist — skip
        log.info(f"━━━ Step 3: No odds updates, using existing predictions ━━━")

    # Step 4: Frontend
    step_frontend(pred_path, args.date)


if __name__ == "__main__":
    main()
