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
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

from sqlalchemy import text

from scraper.db import get_session
from scraper.netkeiba import scrape_race_list, scrape_race, save_race_to_db
from scraper.odds_watcher import fetch_win_odds, fetch_exotic_odds, save_odds_snapshot
from models.predict_final import predict_with_filters
from models.paddock_scorer import score_race_paddock
from results.build_data_json import build_data_json
from notifications.dispatcher import notify_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# Odds fetch window: how far ahead (min) to look for upcoming races
ODDS_WINDOW_MIN = 2880   # fetch for races starting within 48 hours (temporary for pre-race predictions)
ODDS_FINISHED_GRACE = 5  # skip races finished more than 5 min ago


def step_scrape(date: str, force: bool = False) -> list[str]:
    """Step 1: Discover and scrape all races for the date. If force=True, re-scrapes and updates all races."""
    date_compact = date.replace("-", "")
    log.info(f"━━━ Step 1: Scraping races for {date} (force={force}) ━━━")

    race_ids = scrape_race_list(date_compact)
    if not race_ids:
        # Fallback: check if races already exist in DB
        with get_session() as session:
            rows = session.execute(
                text("SELECT netkeiba_id FROM races WHERE date = :d ORDER BY course_id, race_number"),
                {"d": date},
            ).fetchall()
        if rows:
            race_ids = [r.netkeiba_id for r in rows]
            log.info(f"Race list empty but found {len(race_ids)} races in DB — using those")
        else:
            log.error(f"No races found for {date}. Is the date correct?")
            return []

    log.info(f"Found {len(race_ids)} race IDs")

    # Check which are already in DB
    with get_session() as session:
        existing = session.execute(
            text("SELECT netkeiba_id FROM races WHERE date = :d"),
            {"d": date},
        ).fetchall()
    existing_ids = {r.netkeiba_id for r in existing}

    target_ids = race_ids if force else [rid for rid in race_ids if rid not in existing_ids]
    if not target_ids:
        log.info(f"All {len(race_ids)} races already in DB — skipping scrape")
    else:
        log.info(f"Scraping {len(target_ids)} races ({len(existing_ids)} already in DB, force={force})...")
        for i, rid in enumerate(target_ids, 1):
            log.info(f"  [{i}/{len(target_ids)}] {rid}")
            try:
                race_data = scrape_race(rid)
                if race_data:
                    race_obj = race_data[0] if isinstance(race_data, tuple) else race_data
                    if not race_obj.date:
                        race_obj.date = str(date)
                    save_race_to_db(race_data, force=force)
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
                FROM races WHERE date = :d
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

        # post_time is a time object from DB or string (SQLite)
        race_date = datetime.strptime(date, "%Y-%m-%d").date()
        
        post_time_obj = info.post_time
        if isinstance(post_time_obj, str):
            from datetime import time as dt_time
            parts = post_time_obj.split(':')
            post_time_obj = dt_time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
            
        post_dt = datetime.combine(race_date, post_time_obj, tzinfo=JST)
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

    # Sort upcoming_ids chronologically and take only the next 3 races
    from datetime import time as dt_time
    upcoming_ids.sort(key=lambda nk_id: nk_to_info.get(nk_id).post_time if nk_to_info.get(nk_id) and nk_to_info.get(nk_id).post_time else dt_time.max)
    upcoming_ids = upcoming_ids[:3]

    log.info(f"  Targeting next {len(upcoming_ids)} races chronologically "
             f"(past: {skipped_past}, future: {skipped_future}, no_time: {no_time})")

    updated_db_ids = []
    for i, nk_id in enumerate(upcoming_ids, 1):
        db_id = nk_to_db.get(nk_id)
        if not db_id:
            continue

        info = nk_to_info.get(nk_id)
        if info and info.post_time:
            time_str = info.post_time if isinstance(info.post_time, str) else info.post_time.strftime('%H:%M')
            if len(time_str) > 5:
                time_str = time_str[:5]
        else:
            time_str = "??:??"

        odds = fetch_win_odds(nk_id)
        if not odds:
            log.warning(f"  [{i}/{len(upcoming_ids)}] R{info.race_number if info else '?'} ({time_str}): no odds")
            time.sleep(0.5)
            continue

        official_dt = odds[0].get("official_datetime") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_session() as session:
            session.execute(
                text("UPDATE races SET odds_updated_at = :dt WHERE id = :race_id"),
                {"dt": official_dt, "race_id": db_id},
            )
            count = 0
            for o in odds:
                pp = int(o["combination"])
                result = session.execute(
                    text("""
                        UPDATE entries
                        SET odds_win = :odds
                        WHERE race_id = :race_id AND post_position = :pp
                    """),
                    {"odds": o["odds_value"], "race_id": db_id, "pp": pp},
                )
                count += result.rowcount
            session.commit()

        # Also fetch top exotic pools (quinella, wide, exacta, trio)
        try:
            exotics = fetch_exotic_odds(nk_id, bet_types=["quinella", "wide", "exacta", "trio"])
            if exotics:
                save_odds_snapshot(nk_id, exotics)
        except Exception as e:
            log.warning(f"  [{i}/{len(upcoming_ids)}] R{info.race_number if info else '?'}: exotic fetch failed ({e})")

        log.info(f"  [{i}/{len(upcoming_ids)}] R{info.race_number if info else '?'} "
                 f"({time_str}): {len(odds)} horses updated (with exotics)")
        updated_db_ids.append(db_id)
        time.sleep(0.5)

    log.info(f"✅ Updated odds for {len(updated_db_ids)} races")
    return updated_db_ids


def step_paddock(date: str) -> int:
    """
    Step 2.5: Score unscored paddock comments using Gemini NLP.

    Finds any paddock comments in the DB for today's races that
    haven't been scored yet, and runs them through Gemini.

    Returns:
        Total number of comments scored.
    """
    log.info("━━━ Step 2.5: Paddock NLP scoring ━━━")

    with get_session() as session:
        races = session.execute(
            text("SELECT id FROM races WHERE date = :d ORDER BY course_id, race_number"),
            {"d": date},
        ).fetchall()

    if not races:
        log.info("  No races found")
        return 0

    total_scored = 0
    for r in races:
        scored = score_race_paddock(r.id)
        total_scored += scored

    if total_scored > 0:
        log.info(f"✅ Scored {total_scored} paddock comments across {len(races)} races")
    else:
        log.info("  No unscored paddock comments found")

    return total_scored


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
            text("SELECT id FROM races WHERE date = :d ORDER BY course_id, race_number"),
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

    import json
    from pathlib import Path
    from models.train import MODELS_DIR

    meta_path = MODELS_DIR / version / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    from models.predict_final import predict_with_filters

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
                
        # Re-evaluate value bets for kept predictions in case EV threshold or odds filters changed
        for p in kept:
            ev = p.get("ev", 0)
            odds = p.get("odds", 0)
            p["is_value_bet"] = (ev >= ev_threshold) and (min_odds <= odds <= max_odds)

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
        if "is_value_bet" in df.columns:
            value_bets = df[df["is_value_bet"]]
        elif "is_value" in df.columns:
            value_bets = df[df["is_value"]]
            df = df.rename(columns={"is_value": "is_value_bet"})
        else:
            value_bets = pd.DataFrame()

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

    build_data_json([predictions_path], output="results/data.json")

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
            r'<span class="date-value">.*?</span>',
            f'<span class="date-value">{date_jp}</span>',
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


def step_notify(predictions_path: str, date: str, version: str,
                ev_threshold: float):
    """
    Step 6: Send WhatsApp/Telegram notification with value bet summary.

    Sends a rich message containing all value bets with the information
    needed to place bets at the track.
    """
    log.info("━━━ Step 6: Sending notifications ━━━")

    # Load predictions
    pred_file = Path(predictions_path)
    if not pred_file.exists():
        log.warning(f"Predictions file not found: {predictions_path}")
        return

    with open(pred_file) as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    value_bets = [p for p in predictions
                  if p.get("is_value_bet") or p.get("is_value")]

    # Build race info lookup from DB
    with get_session() as session:
        races = session.execute(
            text("""
                SELECT r.id, r.race_number, r.course_id, r.distance, r.surface,
                       r.post_time, r.race_name_jp
                FROM races r WHERE r.date = :d
                ORDER BY r.course_id, r.race_number
            """),
            {"d": date},
        ).fetchall()
    race_map = {r.id: r for r in races}

    COURSE_NAMES = {
        1: 'Sapporo', 2: 'Hakodate', 3: 'Fukushima', 4: 'Niigata',
        5: 'Tokyo', 6: 'Nakayama', 7: 'Chukyo', 8: 'Kyoto',
        9: 'Hanshin', 10: 'Kokura',
    }

    # Header
    dt = datetime.strptime(date, "%Y-%m-%d")
    weekday_jp = ["月", "火", "水", "木", "金", "土", "日"][dt.weekday()]
    header = (
        f"🏇 *UmaEdge — {date} ({weekday_jp})*\n"
        f"Model: {version} | EV ≥ {ev_threshold:.0%}\n"
        f"Races: {len(races)} | Value bets: {len(value_bets)}\n"
        f"{'─' * 28}"
    )

    if not value_bets:
        msg = f"{header}\n\n❌ No value bets found today."
        sent = notify_message(msg)
        if sent:
            log.info(f"✅ Notification sent ({sent} backend(s))")
        else:
            log.info("  No notification backends configured")
        return

    # Sort by race_id to group by race
    value_bets.sort(key=lambda p: (p.get("race_id", 0), -p.get("ev", 0)))

    # Look up post_position + draw from DB for each value bet entry
    entry_ids = [b.get("entry_id") for b in value_bets if b.get("entry_id")]
    entry_pp_map = {}
    if entry_ids:
        with get_session() as session:
            ids_str = ",".join(map(str, entry_ids))
            rows = session.execute(
                text(f"SELECT id, post_position, draw FROM entries WHERE id IN ({ids_str})")
            ).fetchall()
        entry_pp_map = {r.id: (r.post_position, r.draw) for r in rows}

    # Build per-bet lines
    bet_lines = []
    current_race_id = None
    for bet in value_bets:
        race_id = bet.get("race_id")
        race = race_map.get(race_id)

        # Race header (once per race)
        if race_id != current_race_id:
            current_race_id = race_id
            if race:
                venue = COURSE_NAMES.get(race.course_id, "?")
                if race.post_time:
                    post = race.post_time[:5] if isinstance(race.post_time, str) else race.post_time.strftime("%H:%M")
                else:
                    post = "--:--"
                surface = (race.surface or "?").upper()
                dist = race.distance or "?"
                race_name = race.race_name_jp or ""
                bet_lines.append(
                    f"\n📍 *R{race.race_number} {venue}* {post}\n"
                    f"   {surface} {dist}m {race_name}"
                )
            else:
                bet_lines.append(f"\n📍 *Race {race_id}*")

        # Horse line — look up post position from DB
        horse = bet.get("horse_name", "?")
        odds = bet.get("odds", 0)
        prob = bet.get("combined_win_prob", bet.get("combined_prob", 0))
        ev_pct = bet.get("ev", 0)
        entry_id = bet.get("entry_id")
        pp_info = entry_pp_map.get(entry_id, (None, None))
        pp = pp_info[0] or "?"
        draw = pp_info[1] or ""
        draw_str = f" (枠{draw})" if draw else ""

        bet_lines.append(
            f"  🐴 #{pp}{draw_str} *{horse}*\n"
            f"     Odds: {odds:.1f}x | EV: {ev_pct:+.0%}\n"
            f"     Win%: {prob:.1%}\n"
            f"     💴 ¥1,000"
        )

    # Footer
    total_stake = len(value_bets) * 1000
    footer = (
        f"\n{'─' * 28}\n"
        f"💰 {len(value_bets)} bets × ¥1,000 = ¥{total_stake:,}\n"
        f"Good luck! 🍀"
    )

    # CallMeBot truncates around ~1000 chars; use 800 to be safe
    MAX_LEN = 800

    # Always send as chunked messages for reliability
    msgs = [header]
    chunk = ""
    for line in bet_lines:
        if len(chunk) + len(line) + 1 > MAX_LEN:
            if chunk:
                msgs.append(chunk)
            chunk = line
        else:
            chunk += "\n" + line
    if chunk:
        msgs.append(chunk)
    msgs.append(footer)

    sent = 0
    for i, msg in enumerate(msgs):
        if i > 0:
            time.sleep(2)  # rate-limit between sends
        result = notify_message(msg)
        if result:
            sent += 1

    if sent:
        log.info(f"✅ Notifications sent ({sent} message(s))")
    else:
        log.info("  No notification backends configured")


def step_results(date: str, race_ids: list[str] = None, force: bool = False) -> list[int]:
    """
    Step 5: Scrape results for finished races.
    Updates entries with finish_pos, time, last_3f, final odds.
    Returns list of DB race IDs that were synced.
    """
    JST = timezone(timedelta(hours=9))
    now_jst = datetime.now(JST)
    log.info(f"━━━ Step 5: Collecting results for {date} (JST: {now_jst.strftime('%H:%M')}, force={force}) ━━━")

    # Find races for this date
    with get_session() as session:
        races = session.execute(
            text("""
                SELECT id, netkeiba_id, race_number, post_time
                FROM races WHERE date = :d
                ORDER BY course_id, race_number
            """),
            {"d": date},
        ).fetchall()

    if not races:
        log.info(f"  No races found in DB for {date}")
        return []

    finished_ids = []
    try:
        race_date = datetime.strptime(date, "%Y-%m-%d").date()
        is_past_day = race_date < datetime.now(JST).date()
    except Exception:
        is_past_day = False

    for r in races:
        if force or is_past_day:
            # Check if missing finish positions or force
            with get_session() as session:
                counts = session.execute(
                    text("""
                        SELECT COUNT(*) as total,
                               COUNT(finish_pos) as has_result
                        FROM entries
                        WHERE race_id = :rid
                    """),
                    {"rid": r.id},
                ).fetchone()
            if force or (counts and counts.has_result < counts.total):
                finished_ids.append(r)
            continue

        if not r.post_time:
            continue
        post_time_val = r.post_time
        if isinstance(post_time_val, str):
            post_time_str = post_time_val.split('.')[0]
            if post_time_str.count(':') == 1:
                post_time_str += ":00"
            post_time_val = datetime.strptime(post_time_str, "%H:%M:%S").time()
        post_dt = datetime.combine(race_date, post_time_val, tzinfo=JST)
        mins_since = (now_jst - post_dt).total_seconds() / 60
        if mins_since > 10:
            # Check if we already have COMPLETE results (all entries have finish_pos)
            with get_session() as session:
                counts = session.execute(
                    text("""
                        SELECT COUNT(*) as total,
                               COUNT(finish_pos) as has_result
                        FROM entries
                        WHERE race_id = :rid
                    """),
                    {"rid": r.id},
                ).fetchone()
            if counts.has_result < counts.total:
                finished_ids.append(r)

    if not finished_ids:
        log.info("  No new results to collect")
        return []

    log.info(f"  Scraping results concurrently for {len(finished_ids)} finished races")

    def _sync_single_race(race_row):
        try:
            race_data = scrape_race(race_row.netkeiba_id, results_only=True)
            if not race_data or not race_data.entries:
                return None

            has_finish = any(e.finish_pos is not None for e in race_data.entries)
            if not has_finish:
                return None

            with get_session() as session:
                for entry in race_data.entries:
                    if entry.finish_pos is not None:
                        entry_row = session.execute(
                            text("""
                                UPDATE entries
                                SET finish_pos = :fp, time_secs = :ts,
                                    last_3f_secs = :l3f, corner_positions = :cp,
                                    odds_win = COALESCE(:odds, odds_win),
                                    popularity = COALESCE(:pop, popularity)
                                WHERE race_id = :rid AND post_position = :pp
                                RETURNING id
                            """),
                            {
                                "fp": entry.finish_pos,
                                "ts": entry.time_secs,
                                "l3f": entry.last_3f_secs,
                                "cp": entry.corner_positions,
                                "odds": entry.odds_win,
                                "pop": entry.popularity,
                                "rid": race_row.id,
                                "pp": entry.post_position,
                            },
                        ).fetchone()
                        if entry_row:
                            entry_id = entry_row[0]
                            session.execute(
                                text("""
                                    INSERT INTO results (
                                        entry_id, finish_pos, margin, time_secs,
                                        last_3f_secs, corner_positions
                                    ) VALUES (
                                        :entry_id, :finish_pos, :margin, :time_secs,
                                        :last_3f, :corners
                                    )
                                    ON CONFLICT (entry_id) DO UPDATE SET
                                        finish_pos = COALESCE(EXCLUDED.finish_pos, results.finish_pos),
                                        margin = COALESCE(EXCLUDED.margin, results.margin),
                                        time_secs = COALESCE(EXCLUDED.time_secs, results.time_secs),
                                        last_3f_secs = COALESCE(EXCLUDED.last_3f_secs, results.last_3f_secs),
                                        corner_positions = COALESCE(EXCLUDED.corner_positions, results.corner_positions)
                                """),
                                {
                                    "entry_id": entry_id,
                                    "finish_pos": entry.finish_pos,
                                    "margin": getattr(entry, "margin", None),
                                    "time_secs": entry.time_secs,
                                    "last_3f": entry.last_3f_secs,
                                    "corners": entry.corner_positions,
                                },
                            )
                session.commit()
            log.info(f"    ✅ Results saved for R{race_row.race_number} ({race_row.netkeiba_id})")
            return race_row.id
        except Exception as e:
            log.warning(f"    ⚠️ Failed R{race_row.race_number} ({race_row.netkeiba_id}): {e}")
            return None

    workers = min(8, max(1, len(finished_ids)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_sync_single_race, finished_ids))

    synced_db_ids = [rid for rid in results if rid is not None]
    log.info(f"✅ Collected results for {len(synced_db_ids)} races")

    if synced_db_ids:
        log.info("  Updating EWMA Ability Ratings (Elo) for completed races...")
        try:
            from models.ability_rating import AbilityRatingEngine
            engine = AbilityRatingEngine()
            engine.update_for_date(date)
        except Exception as e:
            log.error(f"  ⚠️ Failed to update ability ratings: {e}")

    return synced_db_ids


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
    parser.add_argument("--version", default="20260604_223536", help="Model version (default: 20260604_223536)")
    parser.add_argument("--ev-threshold", type=float, default=0.15, help="Min EV for value bet (default: 15%%)")
    parser.add_argument("--max-odds", type=float, default=20.0, help="Max odds filter (default: 20)")
    parser.add_argument("--min-odds", type=float, default=2.0, help="Min odds filter (default: 2.0)")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip race scraping (already in DB)")
    parser.add_argument("--force-scrape", action="store_true", help="Force re-scraping and updating of all races even if already in DB")
    parser.add_argument("--skip-odds", action="store_true", help="Skip odds fetching")
    parser.add_argument("--skip-predict", action="store_true", help="Skip prediction (reuse existing)")
    parser.add_argument("--skip-paddock", action="store_true", help="Skip paddock NLP scoring")
    parser.add_argument("--skip-notify", action="store_true", help="Skip batch notification (use race_watcher.py instead)")
    parser.add_argument("--force-predict", action="store_true", help="Force prediction for all races")
    args = parser.parse_args()

    log.info(f"🏇 UmaEdge Pipeline — {args.date}")
    log.info(f"   Model: {args.version} | EV threshold: {args.ev_threshold:.0%} | "
             f"Odds: {args.min_odds}-{args.max_odds}x")

    # Step 1: Scrape
    if args.skip_scrape:
        log.info("━━━ Step 1: Scraping → SKIPPED ━━━")
        with get_session() as session:
            rows = session.execute(
                text("SELECT netkeiba_id FROM races WHERE date = :d ORDER BY course_id, race_number"),
                {"d": args.date},
            ).fetchall()
        race_ids = [r.netkeiba_id for r in rows]
        log.info(f"   Found {len(race_ids)} races in DB")
    else:
        race_ids = step_scrape(args.date, force=args.force_scrape)

    if not race_ids:
        log.error("No races found — aborting")
        sys.exit(1)

    # Step 2: Odds (returns list of updated DB race IDs)
    updated_db_ids = []
    if args.skip_odds:
        log.info("━━━ Step 2: Odds → SKIPPED ━━━")
    else:
        updated_db_ids = step_odds(args.date, race_ids)

    # Step 2.5: Paddock NLP
    if args.skip_paddock:
        log.info("━━━ Step 2.5: Paddock → SKIPPED ━━━")
    else:
        pass # step_paddock(args.date)

    # Step 3: Predict
    pred_path = f"results/predictions_{args.date}.json"
    if args.skip_predict:
        if Path(pred_path).exists():
            log.info(f"━━━ Step 3: Predict → SKIPPED (using {pred_path}) ━━━")
        else:
            log.info(f"━━━ Step 3: Predict → SKIPPED (no existing JSON, scrape-only mode) ━━━")
            pred_path = None
    elif args.force_predict or not Path(pred_path).exists():
        # First run or forced: predict all
        pred_path = step_predict(args.date, args.version, args.ev_threshold,
                                 args.max_odds, args.min_odds)
    elif updated_db_ids:
        # Incremental: only re-predict races with freshly updated odds
        pred_path = step_predict(args.date, args.version, args.ev_threshold,
                                 args.max_odds, args.min_odds,
                                 race_db_ids=updated_db_ids)
    else:
        # No odds updated and predictions exist — skip
        log.info(f"━━━ Step 3: No odds updates, using existing predictions ━━━")

    # Step 4: Frontend
    if pred_path:
        step_frontend(pred_path, args.date)
    else:
        log.info("━━━ Step 4: Frontend → SKIPPED (no predictions yet) ━━━")

    # Step 5: Collect results for finished races
    step_results(args.date, race_ids)

    # Step 6: Send notifications
    if args.skip_notify:
        log.info("━━━ Step 6: Notifications → SKIPPED ━━━")
    else:
        step_notify(pred_path, args.date, args.version, args.ev_threshold)


if __name__ == "__main__":
    main()
