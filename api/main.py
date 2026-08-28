"""
UmaEdge — FastAPI Backend API.

Serves race data, model predictions, bankroll metrics,
backtest results, and odds time-series from the local SQLite database.
"""

import os
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from scraper.db import get_session

load_dotenv()

# Active model version — keep in sync with pipeline.py / raceday.yml
MODEL_VERSION = os.getenv("MODEL_VERSION", "retrain_20260822_2143")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="UmaEdge API", version="1.0.0")

# CORS: read from env or fall back to localhost
_cors_origins_str = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001,http://localhost:8080,https://umaedge.vercel.app",
)
_cors_origins = [o.strip() for o in _cors_origins_str.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COURSE_NAMES = {
    1: "Sapporo", 2: "Hakodate", 3: "Fukushima", 4: "Niigata",
    5: "Tokyo", 6: "Nakayama", 7: "Chukyo", 8: "Kyoto",
    9: "Hanshin", 10: "Kokura",
}

# TEXT-stored numeric columns in the predictions table
# (legacy schema: these were added as TEXT but should be floats)
_PRED_NUMERIC_COLS = (
    "edge", "fair_odds", "ability_rating",
    "condition_fit", "bounce_risk",
)


def _coerce_prediction(d: dict) -> dict:
    """Coerce TEXT-stored numeric fields in a prediction row to float."""
    for col in _PRED_NUMERIC_COLS:
        if col in d and isinstance(d[col], str):
            try:
                d[col] = float(d[col])
            except (ValueError, TypeError):
                d[col] = None
    return d


# ===================================================================
# 1. RACES
# ===================================================================

@app.get("/api/races")
def list_races(
    date_str: Optional[str] = Query(None, alias="date", description="Filter by date (YYYY-MM-DD)"),
    upcoming: bool = Query(False, description="If true, only today and future races"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    course: Optional[str] = None,
):
    """List races with optional filters."""
    conditions = []
    params: dict = {}

    if date_str:
        conditions.append("r.date = :date")
        params["date"] = date_str
    elif upcoming:
        conditions.append("r.date >= :today")
        params["today"] = date.today().isoformat()

    if course:
        conditions.append("c.name = :course")
        params["course"] = course

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with get_session() as s:
        rows = s.execute(text(f"""
            SELECT r.id, r.netkeiba_id, r.date, r.race_number, r.race_name, r.race_name_jp,
                   r.distance, r.surface, r.going, r.grade, r.post_time, r.course_id,
                   c.name AS course_name,
                   -- Best value bet for this race (highest EV)
                   vb_best.ev          AS bet_ev,
                   vb_best.model_prob  AS bet_model_prob,
                   h_bet.name_jp       AS bet_horse,
                   e_bet.odds_win      AS bet_odds,
                   e_bet.post_position AS bet_post_position,
                   -- Whether any predictions have been run for this race
                   CASE WHEN pred_count.n > 0 THEN 1 ELSE 0 END AS has_predictions
            FROM races r
            LEFT JOIN courses c ON c.id = r.course_id
            LEFT JOIN (
                SELECT race_id, entry_id, ev, model_prob,
                       ROW_NUMBER() OVER (PARTITION BY race_id ORDER BY ev DESC) AS rn
                FROM value_bets
            ) vb_best ON vb_best.race_id = r.id AND vb_best.rn = 1
            LEFT JOIN entries e_bet ON e_bet.id = vb_best.entry_id
            LEFT JOIN horses  h_bet ON h_bet.id = e_bet.horse_id
            LEFT JOIN (
                SELECT race_id, COUNT(*) AS n FROM predictions GROUP BY race_id
            ) pred_count ON pred_count.race_id = r.id
            {where}
            ORDER BY r.date DESC, r.course_id, r.race_number
            LIMIT :limit OFFSET :offset
        """), {**params, "limit": limit, "offset": offset}).fetchall()

    races = []
    for r in rows:
        d = dict(r._mapping)
        d["has_bet"] = d["bet_ev"] is not None
        d["has_predictions"] = bool(d.get("has_predictions"))
        races.append(d)
    return {"races": races, "count": len(races)}


@app.get("/api/races/{race_id}")
def get_race(race_id: int):
    """Full race detail: race info + entries (with horses/jockeys) + predictions."""
    with get_session() as s:
        race = s.execute(text("""
            SELECT r.*, c.name AS course_name, c.name_jp AS course_name_jp
            FROM races r
            LEFT JOIN courses c ON c.id = r.course_id
            WHERE r.id = :id
        """), {"id": race_id}).fetchone()

        if not race:
            raise HTTPException(404, f"Race {race_id} not found")

        entries = s.execute(text("""
            SELECT e.id, e.race_id, e.post_position, e.draw, e.weight_carried,
                   e.horse_weight, e.horse_weight_change, e.odds_win, e.popularity,
                   COALESCE(e.finish_pos, res.finish_pos) AS finish_pos,
                   COALESCE(e.time_secs, res.time_secs) AS time_secs,
                   COALESCE(e.last_3f_secs, res.last_3f_secs) AS last_3f_secs,
                   COALESCE(e.corner_positions, res.corner_positions) AS corner_positions,
                   COALESCE(e.margin, res.margin) AS margin,
                   h.name AS horse_name, h.name_jp AS horse_name_jp, h.sex, h.birth_year,
                   j.name AS jockey_name, j.name_jp AS jockey_name_jp,
                   t.name AS trainer_name
            FROM entries e
            LEFT JOIN results res ON res.entry_id = e.id
            LEFT JOIN horses h ON h.id = e.horse_id
            LEFT JOIN jockeys j ON j.id = e.jockey_id
            LEFT JOIN trainers t ON t.id = (
                SELECT trainer_id FROM entries WHERE id = e.id LIMIT 1
            )
            WHERE e.race_id = :race_id
            ORDER BY e.post_position
        """), {"race_id": race_id}).fetchall()

        predictions = s.execute(text("""
            SELECT p.*, e.post_position, h.name_jp AS horse_name_jp
            FROM predictions p
            LEFT JOIN entries e ON e.id = p.entry_id
            LEFT JOIN horses h ON h.id = e.horse_id
            WHERE p.race_id = :race_id
              AND p.id IN (
                  -- Keep only the most-recent prediction per entry (latest created_at)
                  SELECT id FROM (
                      SELECT id, ROW_NUMBER() OVER (
                          PARTITION BY entry_id ORDER BY created_at DESC, id DESC
                      ) AS rn
                      FROM predictions
                      WHERE race_id = :race_id
                  ) t WHERE rn = 1
              )
            ORDER BY p.win_prob DESC
        """), {"race_id": race_id}).fetchall()

        value_bets = s.execute(text("""
            SELECT vb.*, e.post_position, h.name_jp AS horse_name_jp
            FROM value_bets vb
            LEFT JOIN entries e ON e.id = vb.entry_id
            LEFT JOIN horses h ON h.id = e.horse_id
            WHERE vb.race_id = :race_id
            ORDER BY vb.ev DESC
        """), {"race_id": race_id}).fetchall()

    return {
        "race": dict(race._mapping),
        "entries": [dict(r._mapping) for r in entries],
        "predictions": [_coerce_prediction(dict(r._mapping)) for r in predictions],
        "value_bets": [dict(r._mapping) for r in value_bets],
    }


@app.get("/api/races/{race_id}/betting-analysis")
def get_race_betting_analysis(
    race_id: int,
    budget: int = Query(1000, ge=100, le=1000000, description="Total betting budget in yen"),
    mode: str = Query("auto", description="Staking strategy mode: 'auto', 'hybrid', 'pure_win', or 'dutching'"),
    model_version: Optional[str] = Query(None, description="Model version for predictions"),
):
    """
    Betting analysis, fair pricing breakdown, and multi-ticket portfolio staking
    for a race as specified in docs/BETTING_ANALYSIS_LOGIC.md.
    """
    from models.betting_engine import analyze_race_betting

    with get_session() as s:
        try:
            return analyze_race_betting(
                race_id=race_id,
                budget=budget,
                strategy_mode=mode,
                model_version=model_version,
                session=s,
            )
        except ValueError as e:
            msg = str(e)
            if "not found" in msg.lower() or "no entries" in msg.lower():
                raise HTTPException(404, msg)
            raise HTTPException(400, msg)
        except Exception as e:
            raise HTTPException(500, f"Betting analysis failed: {str(e)}")



# ===================================================================
# 2. PREDICTIONS
# ===================================================================

@app.get("/api/predictions/{race_id}")
def get_predictions(race_id: int, model_version: str = None):
    """Model predictions + value bets for a race."""
    mv = model_version or MODEL_VERSION
    with get_session() as s:
        q = """
            SELECT p.*, e.post_position, e.draw, e.odds_win,
                   h.name AS horse_name, h.name_jp AS horse_name_jp,
                   j.name AS jockey_name
            FROM predictions p
            LEFT JOIN entries e ON e.id = p.entry_id
            LEFT JOIN horses h ON h.id = e.horse_id
            LEFT JOIN jockeys j ON j.id = e.jockey_id
            WHERE p.race_id = :race_id AND p.model_version = :mv
        """
        predictions = s.execute(text(q), {"race_id": race_id, "mv": mv}).fetchall()

        # Fall back to any version if the current model has no predictions yet
        if not predictions:
            predictions = s.execute(text("""
                SELECT p.*, e.post_position, e.draw, e.odds_win,
                       h.name AS horse_name, h.name_jp AS horse_name_jp,
                       j.name AS jockey_name
                FROM predictions p
                LEFT JOIN entries e ON e.id = p.entry_id
                LEFT JOIN horses h ON h.id = e.horse_id
                LEFT JOIN jockeys j ON j.id = e.jockey_id
                WHERE p.race_id = :race_id
                ORDER BY p.created_at DESC
            """), {"race_id": race_id}).fetchall()

        value_bets = s.execute(text("""
            SELECT vb.*, e.post_position, h.name_jp AS horse_name_jp
            FROM value_bets vb
            LEFT JOIN entries e ON e.id = vb.entry_id
            LEFT JOIN horses h ON h.id = e.horse_id
            WHERE vb.race_id = :race_id
            ORDER BY vb.ev DESC
        """), {"race_id": race_id}).fetchall()

    return {
        "predictions": [_coerce_prediction(dict(r._mapping)) for r in predictions],
        "value_bets": [dict(r._mapping) for r in value_bets],
    }


# ===================================================================
# 3. ODDS
# ===================================================================

@app.get("/api/odds/{race_id}")
def get_odds(race_id: int, bet_type: str = "win"):
    """Odds time-series for a race."""
    with get_session() as s:
        rows = s.execute(text("""
            SELECT * FROM odds_snapshots
            WHERE race_id = :race_id AND bet_type = :bet_type
            ORDER BY captured_at
        """), {"race_id": race_id, "bet_type": bet_type}).fetchall()

    return {"odds": [dict(r._mapping) for r in rows]}


# ===================================================================
# 4. BACKTEST
# ===================================================================

@app.get("/api/backtest")
def get_backtest(
    model_version: str = Query("latest"),
    grade: Optional[str] = None,
    course: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    """
    Backtest summary: aggregate metrics + P&L curve.
    Reads from value_bets + entries + races in the local DB.
    """
    conditions = ["vb.ev > 0"]
    params: dict = {}

    if model_version != "latest":
        conditions.append("vb.model_version = :mv")
        params["mv"] = model_version
    if grade:
        conditions.append("r.grade = :grade")
        params["grade"] = grade
    if date_from:
        conditions.append("r.date >= :date_from")
        params["date_from"] = date_from
    if date_to:
        conditions.append("r.date <= :date_to")
        params["date_to"] = date_to

    where = "WHERE " + " AND ".join(conditions)

    with get_session() as s:
        if course:
            rows = s.execute(text(f"""
                SELECT vb.ev, vb.model_prob, vb.market_prob, vb.recommended_stake,
                       e.odds_win, COALESCE(e.finish_pos, res.finish_pos) AS finish_pos,
                       r.date, r.race_name, r.grade, r.course_id,
                       c.name AS course_name,
                       h.name AS horse_name
                FROM value_bets vb
                JOIN entries e ON e.id = vb.entry_id
                LEFT JOIN results res ON res.entry_id = e.id
                JOIN races r ON r.id = vb.race_id
                LEFT JOIN courses c ON c.id = r.course_id
                LEFT JOIN horses h ON h.id = e.horse_id
                {where} AND c.name = :course
                ORDER BY r.date
            """), {**params, "course": course}).fetchall()
        else:
            rows = s.execute(text(f"""
                SELECT vb.ev, vb.model_prob, vb.market_prob, vb.recommended_stake,
                       e.odds_win, COALESCE(e.finish_pos, res.finish_pos) AS finish_pos,
                       r.date, r.race_name, r.grade, r.course_id,
                       h.name AS horse_name
                FROM value_bets vb
                JOIN entries e ON e.id = vb.entry_id
                LEFT JOIN results res ON res.entry_id = e.id
                JOIN races r ON r.id = vb.race_id
                LEFT JOIN horses h ON h.id = e.horse_id
                {where}
                ORDER BY r.date
            """), params).fetchall()

    bets = []
    for row in rows:
        r = dict(row._mapping)
        odds = r.get("odds_win") or 0
        if odds <= 1:
            continue
        stake = 1000
        won = r.get("finish_pos") == 1
        payout = stake * odds if won else 0
        profit = payout - stake
        bets.append({
            "date": str(r.get("date", "")),
            "race_name": r.get("race_name"),
            "grade": r.get("grade"),
            "horse": r.get("horse_name"),
            "model_prob": round(r.get("model_prob") or 0, 4),
            "market_prob": round(r.get("market_prob") or 0, 4),
            "ev": round(r.get("ev") or 0, 4),
            "odds": odds,
            "won": won,
            "stake": stake,
            "payout": payout,
            "profit": profit,
        })

    total_bets = len(bets)
    wins = sum(1 for b in bets if b["won"])
    total_staked = total_bets * 1000
    total_profit = sum(b["profit"] for b in bets)

    balance = 100_000
    curve = []
    for b in bets:
        balance += b["profit"]
        curve.append({"date": b["date"], "balance": balance, "profit": b["profit"]})

    roi_by_grade: dict = {}
    for b in bets:
        g = b.get("grade") or "Other"
        roi_by_grade.setdefault(g, {"staked": 0, "profit": 0, "count": 0})
        roi_by_grade[g]["staked"] += b["stake"]
        roi_by_grade[g]["profit"] += b["profit"]
        roi_by_grade[g]["count"] += 1
    for g in roi_by_grade:
        s = roi_by_grade[g]
        s["roi"] = round(s["profit"] / s["staked"] * 100, 2) if s["staked"] else 0

    return {
        "total_bets": total_bets,
        "wins": wins,
        "win_rate": round(wins / total_bets * 100, 1) if total_bets else 0,
        "total_staked": total_staked,
        "total_profit": round(total_profit),
        "roi_pct": round(total_profit / total_staked * 100, 2) if total_staked else 0,
        "balance_curve": curve,
        "roi_by_grade": roi_by_grade,
        "bets": bets[:100],
    }


# ===================================================================
# 5. MANIFEST (for the calendar frontend)
# ===================================================================

@app.get("/api/manifest")
def get_manifest():
    """
    Returns a list of all race dates that have predictions in the DB,
    with summary stats per date — used by the frontend calendar.
    """
    with get_session() as s:
        rows = s.execute(text("""
            SELECT
                r.date,
                COUNT(DISTINCT r.id)   AS total_races,
                COUNT(DISTINCT vb_top.race_id) AS total_bets,
                SUM(CASE WHEN COALESCE(e.finish_pos, res.finish_pos) = 1 AND vb_top.race_id IS NOT NULL THEN 1 ELSE 0 END) AS winners
            FROM races r
            LEFT JOIN (
                SELECT race_id, entry_id, model_version,
                       ROW_NUMBER() OVER (
                           PARTITION BY race_id 
                           ORDER BY CASE WHEN model_version = :mv THEN 0 ELSE 1 END, ev DESC, id DESC
                       ) as rn
                FROM value_bets
            ) vb_top ON vb_top.race_id = r.id AND vb_top.rn = 1
            LEFT JOIN entries e ON e.id = vb_top.entry_id
            LEFT JOIN results res ON res.entry_id = e.id
            GROUP BY r.date
            ORDER BY r.date DESC
        """), {"mv": MODEL_VERSION}).fetchall()

    dates = []
    for row in rows:
        r = dict(row._mapping)
        total_bets = r.get("total_bets") or 0
        winners = r.get("winners") or 0
        dates.append({
            "date": str(r["date"]),
            "races": r.get("total_races") or 0,
            "bets": total_bets,
            "winners": winners,
            "strike_rate": round(winners / total_bets * 100, 1) if total_bets else None,
        })

    return {"dates": dates, "model_version": MODEL_VERSION}


# ===================================================================
# 6. NOTIFICATIONS
# ===================================================================

@app.get("/api/notifications/status")
def notification_status():
    """Check which notification backends are configured."""
    from notifications.dispatcher import get_status
    return get_status()


@app.post("/api/notifications/test")
def test_notifications():
    """Send a test message to all configured notification backends."""
    from notifications.dispatcher import notify_message, get_status

    status = get_status()
    if not status["any_configured"]:
        raise HTTPException(
            400,
            "No notification backends configured. "
            "Set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID or "
            "LINE_CHANNEL_TOKEN + LINE_USER_ID in .env",
        )

    sent = notify_message("🏇 UmaEdge test notification — everything is working!")
    return {"sent": sent, "backends": status}


# ===================================================================
# 7. AUTOMATION PIPELINE
# ===================================================================

@app.post("/api/scraper/date/{date_str}")
def scrape_date(
    date_str: str,
    force: bool = Query(False, description="Whether to re-scrape and update existing races in the database"),
):
    """Trigger the scraper pipeline to fetch races for a specific date (YYYY-MM-DD)."""
    try:
        from pipeline import step_scrape
        force_flag = bool(force) if isinstance(force, bool) else (str(force).lower() in ("true", "1", "yes"))
        race_ids = step_scrape(date_str, force=force_flag)
        return {
            "status": "success",
            "message": f"Scraped {len(race_ids)} races for {date_str}",
            "race_ids": race_ids,
            "races_count": len(race_ids),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


@app.post("/api/races/date/{date_str}/predict")
@app.post("/api/races/predict-date/{date_str}")
def predict_date_races(
    date_str: str,
    version: Optional[str] = Query(None, description="Model version to use for predictions"),
    ev_threshold: float = Query(0.08, description="EV threshold for value bets (default: 8% to match EV_PASS)"),
    max_odds: float = Query(20.0, description="Max odds for value bets"),
    min_odds: float = Query(2.0, description="Min odds for value bets"),
    refresh_odds: bool = Query(True, description="Whether to fetch latest live odds first"),
):
    """
    Run model predictions for ALL races on a given date (YYYY-MM-DD),
    persisting predictions and value bets into the database.
    """
    model = version if (isinstance(version, str) and version) else MODEL_VERSION
    ev_thresh = float(ev_threshold) if isinstance(ev_threshold, (int, float)) else 0.08
    max_o = float(max_odds) if isinstance(max_odds, (int, float)) else 20.0
    min_o = float(min_odds) if isinstance(min_odds, (int, float)) else 2.0
    ref_odds = refresh_odds if isinstance(refresh_odds, bool) else True

    # 1. Fetch all races for this date
    with get_session() as s:
        race_rows = s.execute(
            text("""
                SELECT id, netkeiba_id, race_number, course_id
                FROM races
                WHERE date = :date
                ORDER BY course_id, race_number
            """),
            {"date": date_str},
        ).fetchall()

    if not race_rows:
        raise HTTPException(
            status_code=404,
            detail=f"No races found in database for date {date_str}. Please scrape races first."
        )

    race_ids = [r.id for r in race_rows]

    # 2. Optionally refresh live odds
    # Auto-skip for past dates: netkeiba removes live odds pages after race day,
    # so scraping them fails and causes the whole batch predict to show "Retry".
    from datetime import date as _date
    try:
        is_past_date = _date.fromisoformat(date_str) < _date.today()
    except ValueError:
        is_past_date = False
    if is_past_date:
        refresh_odds = False

    updated_odds_count = 0
    if refresh_odds:
        from scraper.odds_watcher import fetch_win_odds
        with get_session() as s:
            for r in race_rows:
                if r.netkeiba_id:
                    odds = fetch_win_odds(r.netkeiba_id)
                    if odds:
                        for o in odds:
                            s.execute(
                                text("""
                                    UPDATE entries
                                    SET odds_win = :odds,
                                        popularity = COALESCE(:pop, popularity)
                                    WHERE race_id = :race_id AND post_position = :pp
                                """),
                                {
                                    "odds": o["odds_value"],
                                    "pop": o.get("popularity"),
                                    "race_id": r.id,
                                    "pp": int(o["combination"]),
                                },
                            )
                        updated_odds_count += len(odds)
            s.commit()

    # 3. Predict with filters across all races
    try:
        from models.predict_final import predict_with_filters
        df = predict_with_filters(
            race_ids=race_ids,
            model_version=model,
            ev_threshold=ev_thresh,
            max_odds=max_o,
            min_odds=min_o,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

    if df is None or df.empty:
        raise HTTPException(status_code=500, detail=f"No predictions generated for date {date_str}")

    # 4. Persist predictions + value bets into DB
    now = datetime.utcnow().isoformat()
    value_bets_count = 0
    races_with_value_bets = set()

    with get_session() as s:
        id_list_str = ",".join(str(rid) for rid in race_ids)
        s.execute(text(f"DELETE FROM predictions WHERE race_id IN ({id_list_str}) AND model_version = :mv"), {"mv": model})
        s.execute(text(f"DELETE FROM value_bets WHERE race_id IN ({id_list_str}) AND model_version = :mv"), {"mv": model})

        for _, row in df.iterrows():
            rid = int(row["race_id"])
            eid = int(row["entry_id"])
            ev_val = float(row["ev"]) if (row["ev"] is not None and not pd.isna(row["ev"])) else None
            s.execute(
                text("""
                    INSERT INTO predictions
                        (race_id, entry_id, model_version, win_prob, edge, created_at)
                    VALUES
                        (:race_id, :entry_id, :mv, :win_prob, :edge, :created_at)
                """),
                {
                    "race_id": rid,
                    "entry_id": eid,
                    "mv": model,
                    "win_prob": float(row["combined_prob"]),
                    "edge": ev_val,
                    "created_at": now,
                },
            )
            if row.get("is_value_bet") and ev_val is not None:
                value_bets_count += 1
                races_with_value_bets.add(rid)
                s.execute(
                    text("""
                        INSERT INTO value_bets
                            (race_id, entry_id, bet_type, model_prob, market_prob,
                             ev, kelly_fraction, recommended_stake, model_version, created_at)
                        VALUES
                            (:race_id, :entry_id, 'win', :model_prob, :market_prob,
                             :ev, :kelly, :stake, :mv, :created_at)
                    """),
                    {
                        "race_id": rid,
                        "entry_id": eid,
                        "model_prob": float(row["combined_prob"]),
                        "market_prob": float(row["market_prob"]) if row["market_prob"] is not None else 0.0,
                        "ev": ev_val,
                        "kelly": float(row["kelly_fraction"]),
                        "stake": int(row["recommended_stake"]),
                        "mv": model,
                        "created_at": now,
                    },
                )
        s.commit()

    return {
        "status": "success",
        "message": f"Successfully predicted {len(race_ids)} races ({len(df)} entries) for {date_str}",
        "date": date_str,
        "model": model,
        "races_predicted": len(race_ids),
        "entries_predicted": len(df),
        "value_bets_count": value_bets_count,
        "races_with_value_bets": sorted(list(races_with_value_bets)),
        "updated_odds_horses": updated_odds_count,
    }


@app.post("/api/races/{race_id}/refresh")
def refresh_race(race_id: int, version: str = None):
    """Refresh live odds and recalculate predictions for a specific race."""
    model = version or MODEL_VERSION

    # 1. Look up race info
    with get_session() as s:
        race_row = s.execute(
            text("SELECT netkeiba_id, date FROM races WHERE id = :id"),
            {"id": race_id},
        ).fetchone()

    if not race_row:
        raise HTTPException(status_code=404, detail="Race not found")

    nk_id = race_row.netkeiba_id
    race_date = str(race_row.date)

    # 2. Fetch new odds
    from scraper.odds_watcher import fetch_win_odds
    odds = fetch_win_odds(nk_id)
    if odds:
        with get_session() as s:
            for o in odds:
                s.execute(
                    text("""
                        UPDATE entries
                        SET odds_win = :odds,
                            popularity = COALESCE(:pop, popularity)
                        WHERE race_id = :race_id AND post_position = :pp
                    """),
                    {
                        "odds": o["odds_value"],
                        "pop": o.get("popularity"),
                        "race_id": race_id,
                        "pp": int(o["combination"]),
                    },
                )
            s.commit()

    # 3. Re-predict using the pipeline's predict_with_filters
    try:
        from models.predict_final import predict_with_filters
        df = predict_with_filters(
            race_ids=[race_id],
            model_version=model,
            ev_threshold=0.08,
            max_odds=20.0,
            min_odds=2.0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="No predictions generated")

    # 4. Persist predictions + value bets to the DB so the frontend can display them
    now = datetime.utcnow().isoformat()
    with get_session() as s:
        # Delete stale predictions and value bets for this race + model version
        s.execute(
            text("DELETE FROM predictions WHERE race_id = :race_id AND model_version = :mv"),
            {"race_id": race_id, "mv": model},
        )
        s.execute(
            text("DELETE FROM value_bets WHERE race_id = :race_id AND model_version = :mv"),
            {"race_id": race_id, "mv": model},
        )
        for _, row in df.iterrows():
            ev_val = float(row["ev"]) if (row["ev"] is not None and not pd.isna(row["ev"])) else None
            s.execute(
                text("""
                    INSERT INTO predictions
                        (race_id, entry_id, model_version, win_prob, edge, created_at)
                    VALUES
                        (:race_id, :entry_id, :mv, :win_prob, :edge, :created_at)
                """),
                {
                    "race_id": int(row["race_id"]),
                    "entry_id": int(row["entry_id"]),
                    "mv": model,
                    "win_prob": float(row["combined_prob"]),
                    "edge": ev_val,
                    "created_at": now,
                },
            )
            if row.get("is_value_bet") and ev_val is not None:
                s.execute(
                    text("""
                        INSERT INTO value_bets
                            (race_id, entry_id, bet_type, model_prob, market_prob,
                             ev, kelly_fraction, recommended_stake, model_version, created_at)
                        VALUES
                            (:race_id, :entry_id, 'win', :model_prob, :market_prob,
                             :ev, :kelly, :stake, :mv, :created_at)
                    """),
                    {
                        "race_id": int(row["race_id"]),
                        "entry_id": int(row["entry_id"]),
                        "model_prob": float(row["combined_prob"]),
                        "market_prob": float(row["market_prob"]) if row["market_prob"] is not None else 0.0,
                        "ev": ev_val,
                        "kelly": float(row["kelly_fraction"]),
                        "stake": int(row["recommended_stake"]),
                        "mv": model,
                        "created_at": now,
                    },
                )
        s.commit()

    return {
        "status": "success",
        "message": f"Odds updated and predictions recalculated ({model})",
        "model": model,
        "updated_horses": len(odds) if odds else 0,
        "predictions": len(df),
    }


@app.post("/api/races/{race_id}/rescrape")
def rescrape_race(race_id: int, version: Optional[str] = Query(None)):
    """Re-scrape a single race card from netkeiba, update entries/jockeys/weights in DB, and re-predict."""
    with get_session() as s:
        race_row = s.execute(
            text("SELECT netkeiba_id, date FROM races WHERE id = :id"),
            {"id": race_id},
        ).fetchone()

    if not race_row:
        raise HTTPException(status_code=404, detail="Race not found")

    nk_id = race_row.netkeiba_id
    from scraper.netkeiba import scrape_race, save_race_to_db
    race_data = scrape_race(nk_id)
    if not race_data:
        raise HTTPException(status_code=502, detail=f"Failed to scrape race {nk_id} from netkeiba")

    race_obj = race_data[0] if isinstance(race_data, tuple) else race_data
    if not race_obj.date:
        race_obj.date = str(race_row.date)
    save_race_to_db(race_data, force=True)

    # Re-evaluate predictions with latest odds
    return refresh_race(race_id=race_id, version=version)


@app.post("/api/results/date/{date_str}")
@app.post("/api/races/date/{date_str}/sync-results")
def sync_date_results(
    date_str: str,
    force: bool = Query(False, description="Whether to re-scrape already recorded results"),
):
    """Scrape and record official race results (finish positions, times, payouts) for finished races."""
    try:
        from pipeline import step_results
        synced_ids = step_results(date_str, force=force)
        return {
            "status": "success",
            "message": f"Synced results for {len(synced_ids)} races on {date_str}",
            "races_synced": len(synced_ids),
            "race_ids": synced_ids,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Result sync failed: {str(e)}")


# ===================================================================
# Health
# ===================================================================

@app.get("/api/health")
def health():
    with get_session() as s:
        race_count = s.execute(text("SELECT COUNT(*) FROM races")).scalar()
    return {
        "status": "ok",
        "service": "umaedge-api",
        "timestamp": datetime.utcnow().isoformat(),
        "db_races": race_count,
    }


# Mount frontend static files so web app is accessible directly
from fastapi.staticfiles import StaticFiles

if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

