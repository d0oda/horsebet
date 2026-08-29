"""
UmaEdge — FastAPI Backend API.

Serves race data, model predictions, bankroll metrics,
backtest results, and odds time-series from the local SQLite database.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r".*",
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
    where_sub = ("WHERE " + " AND ".join([c.replace("r.", "r_sub.").replace("c.", "c_sub.") for c in conditions])) if conditions else ""

    with get_session() as s:
        rows = s.execute(text(f"""
            SELECT r.id, r.netkeiba_id, r.date, r.race_number, r.race_name, r.race_name_jp,
                   r.distance, r.surface, r.going, r.grade, r.post_time, r.course_id,
                   r.odds_updated_at,
                   c.name AS course_name,
                   -- Best value bet for this race (highest EV)
                   vb_best.ev          AS bet_ev,
                   vb_best.model_prob  AS bet_model_prob,
                   h_bet.name_jp       AS bet_horse,
                   e_bet.odds_win      AS bet_odds,
                   e_bet.post_position AS bet_post_position,
                   COALESCE(e_bet.finish_pos, res_bet.finish_pos) AS bet_finish_pos,
                   -- Whether any predictions have been run for this race
                   CASE WHEN pred_count.n > 0 THEN 1 ELSE 0 END AS has_predictions,
                   pred_count.latest_created_at AS pred_created_at,
                   -- Whether official finish results have been recorded for this race
                   CASE WHEN res_count.n > 0 THEN 1 ELSE 0 END AS is_finished
            FROM races r
            LEFT JOIN courses c ON c.id = r.course_id
            LEFT JOIN (
                SELECT race_id, entry_id, ev, model_prob,
                       ROW_NUMBER() OVER (
                           PARTITION BY race_id 
                           ORDER BY CASE WHEN model_version = :mv THEN 0 ELSE 1 END, ev DESC, id DESC
                       ) AS rn
                FROM value_bets
                WHERE race_id IN (
                    SELECT r_sub.id FROM races r_sub
                    LEFT JOIN courses c_sub ON c_sub.id = r_sub.course_id
                    {where_sub}
                )
            ) vb_best ON vb_best.race_id = r.id AND vb_best.rn = 1
            LEFT JOIN entries e_bet ON e_bet.id = vb_best.entry_id
            LEFT JOIN results res_bet ON res_bet.entry_id = e_bet.id
            LEFT JOIN horses  h_bet ON h_bet.id = e_bet.horse_id
            LEFT JOIN (
                SELECT race_id, COUNT(*) AS n, MAX(created_at) AS latest_created_at
                FROM predictions
                WHERE race_id IN (
                    SELECT r_sub.id FROM races r_sub
                    LEFT JOIN courses c_sub ON c_sub.id = r_sub.course_id
                    {where_sub}
                )
                GROUP BY race_id
            ) pred_count ON pred_count.race_id = r.id
            LEFT JOIN (
                SELECT e_res.race_id, COUNT(*) AS n
                FROM entries e_res
                LEFT JOIN results r_res ON r_res.entry_id = e_res.id
                WHERE COALESCE(e_res.finish_pos, r_res.finish_pos) IS NOT NULL
                GROUP BY e_res.race_id
            ) res_count ON res_count.race_id = r.id
            {where}
            ORDER BY r.date DESC, r.course_id, r.race_number
            LIMIT :limit OFFSET :offset
        """), {**params, "limit": limit, "offset": offset, "mv": MODEL_VERSION}).fetchall()

    races = []
    for r in rows:
        d = dict(r._mapping)
        d["has_bet"] = d["bet_ev"] is not None
        d["has_predictions"] = bool(d.get("has_predictions"))
        d["is_finished"] = bool(d.get("is_finished"))
        d["bet_won"] = bool(d["has_bet"] and d.get("bet_finish_pos") == 1)
        d["bet_payout"] = round(d["bet_odds"] * 1000) if (d["bet_won"] and d.get("bet_odds")) else 0
        d["odds_as_of"] = d.get("odds_updated_at") or d.get("pred_created_at")
        races.append(d)
    return {"races": races, "count": len(races)}


@app.get("/api/races/all-time-returns")
@app.get("/api/races/all-time/returns")
def get_races_all_time_returns_route(
    budget_per_race: int = Query(1000, ge=100, le=1000000, description="Base stake budget per race in yen"),
    model_version: Optional[str] = Query(None, description="Model version for predictions"),
    start_month: Optional[str] = Query(None, description="Starting month in YYYY-MM format"),
    end_month: Optional[str] = Query(None, description="Ending month in YYYY-MM format"),
    force_refresh: bool = False,
):
    return get_all_time_returns(
        budget_per_race=budget_per_race,
        model_version=model_version,
        start_month=start_month,
        end_month=end_month,
        force_refresh=force_refresh,
    )


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
                   NULL AS trainer_name
            FROM entries e
            LEFT JOIN results res ON res.entry_id = e.id
            LEFT JOIN horses h ON h.id = e.horse_id
            LEFT JOIN jockeys j ON j.id = e.jockey_id
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

    race_dict = dict(race._mapping)
    latest_pred_time = dict(predictions[0]._mapping).get("created_at") if predictions else None
    race_dict["odds_as_of"] = race_dict.get("odds_updated_at") or latest_pred_time

    return {
        "race": race_dict,
        "odds_as_of": race_dict["odds_as_of"],
        "odds_updated_at": race_dict["odds_as_of"],
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


def _clean_param(val):
    from fastapi.params import Query as QueryParam
    if isinstance(val, QueryParam):
        return val.default if val.default is not ... else None
    return val


_daily_returns_cache = {}
_monthly_returns_cache = {}
_all_time_returns_cache = {}


@app.get("/api/races/date/{date_str}/daily-returns")
def get_daily_returns(
    date_str: str,
    budget_per_race: int = Query(1000, ge=100, le=1000000, description="Base stake budget per race in yen"),
    model_version: Optional[str] = Query(None, description="Model version for predictions"),
    force_refresh: bool = False,
):
    """
    Computes aggregated daily betting performance & returns across all 4 strategy modes
    ('auto', 'pure_win', 'hybrid', 'dutching') for a specific race date.
    """
    budget_val = _clean_param(budget_per_race) or 1000
    mv_val = _clean_param(model_version)
    cache_key = (date_str, budget_val, mv_val)
    if not force_refresh and cache_key in _daily_returns_cache:
        return _daily_returns_cache[cache_key]

    from models.betting_engine import analyze_race_betting

    strategy_meta_map = {
        "auto": {
            "id": "auto",
            "name": "AI Auto (Optimal)",
            "icon": "🧠",
            "tagline": "AI Dynamic Per-Race Optimal Routing",
        },
        "pure_win": {
            "id": "pure_win",
            "name": "Pure Win (Max ROI)",
            "icon": "⚡",
            "tagline": "100% Single Top Value Pick",
        },
        "hybrid": {
            "id": "hybrid",
            "name": "Balanced (Win + Exotics)",
            "icon": "🎯",
            "tagline": "75% Core Win + 25% Elite Exotics",
        },
        "dutching": {
            "id": "dutching",
            "name": "Dual Dutching (Low DD)",
            "icon": "🛡️",
            "tagline": "Dual Value Win Split — Lowest Drawdown",
        },
    }

    with get_session() as s:
        races = s.execute(
            text("SELECT id, race_number, course_id FROM races WHERE date = :d ORDER BY course_id, race_number"),
            {"d": date_str},
        ).fetchall()

        if not races:
            raise HTTPException(404, f"No races found for date {date_str}")

        entries = s.execute(
            text("""
                SELECT e.race_id, e.post_position, COALESCE(e.finish_pos, res.finish_pos) AS finish_pos, e.odds_win
                FROM entries e
                LEFT JOIN results res ON res.entry_id = e.id
                WHERE e.race_id IN (SELECT id FROM races WHERE date = :d)
            """),
            {"d": date_str},
        ).fetchall()

        entry_map = {(e.race_id, e.post_position): (e.finish_pos, e.odds_win or 1.0) for e in entries}
        results_count = sum(1 for e in entries if e.finish_pos is not None)
        has_results = results_count > 0

        modes = ["auto", "pure_win", "hybrid", "dutching"]
        mode_stats = {
            m: {
                "total_staked": 0,
                "total_payout": 0,
                "bets_placed": 0,
                "bets_won": 0,
                "pending_staked": 0,
                "pending_races": 0,
            }
            for m in modes
        }

        for r in races:
            try:
                analysis = analyze_race_betting(
                    race_id=r.id,
                    budget=budget_per_race,
                    strategy_mode="auto",
                    model_version=model_version,
                    session=s,
                    include_all_modes=True,
                )
                results_by_mode = analysis.get("all_strategies_results", {})

                for m in modes:
                    s_res = results_by_mode.get(m, {})
                    if not s_res:
                        continue

                    stk = s_res.get("staked", 0)
                    pay = s_res.get("payout", 0)
                    is_settled = s_res.get("is_settled", False)

                    if not is_settled:
                        if stk > 0:
                            mode_stats[m]["pending_staked"] += stk
                            mode_stats[m]["pending_races"] += 1
                        continue

                    if stk > 0:
                        mode_stats[m]["total_staked"] += stk
                        mode_stats[m]["total_payout"] += pay
                        mode_stats[m]["bets_placed"] += 1
                        if pay > stk:
                            mode_stats[m]["bets_won"] += 1
            except Exception:
                pass

        strategies_summary = {}
        for mode in modes:
            stats = mode_stats[mode]
            total_staked = stats["total_staked"]
            total_payout = stats["total_payout"]
            bets_placed = stats["bets_placed"]
            bets_won = stats["bets_won"]
            profit = total_payout - total_staked
            roi_pct = round((profit / total_staked * 100), 1) if total_staked > 0 else 0.0
            strike_rate = round((bets_won / bets_placed * 100), 1) if bets_placed > 0 else 0.0
            meta = strategy_meta_map.get(mode, {})

            strategies_summary[mode] = {
                "id": mode,
                "name": meta.get("name", mode.title()),
                "icon": meta.get("icon", "📊"),
                "tagline": meta.get("tagline", ""),
                "staked": total_staked,
                "payout": round(total_payout),
                "profit": round(profit),
                "roi_pct": roi_pct,
                "bets_placed": bets_placed,
                "bets_won": bets_won,
                "strike_rate": strike_rate,
                "pending_staked": stats["pending_staked"],
                "pending_races": stats["pending_races"],
            }

        res_obj = {
            "date": date_str,
            "has_results": has_results,
            "total_races": len(races),
            "budget_per_race": budget_per_race,
            "strategies": strategies_summary,
        }
        _daily_returns_cache[cache_key] = res_obj
        return res_obj


@app.get("/api/races/month/{month_str}/monthly-returns")
@app.get("/api/portfolio/monthly")
def get_monthly_returns(
    month_str: Optional[str] = None,
    month: Optional[str] = Query(None, description="Month in YYYY-MM format"),
    budget_per_race: int = Query(1000, ge=100, le=1000000, description="Base stake budget per race in yen"),
    model_version: Optional[str] = Query(None, description="Model version for predictions"),
    force_refresh: bool = False,
):
    """
    Computes aggregated monthly betting performance & returns across all 4 strategy modes
    ('auto', 'pure_win', 'hybrid', 'dutching') for a specific month (YYYY-MM).
    Includes day-by-day progression timeline and available months discovery.
    """
    target_month = _clean_param(month_str) or _clean_param(month)
    budget_val = _clean_param(budget_per_race) or 1000
    mv_val = _clean_param(model_version)

    if not target_month:
        target_month = datetime.now().strftime("%Y-%m")

    if not re.match(r"^\d{4}-\d{2}$", target_month):
        raise HTTPException(400, f"Invalid month format: '{target_month}'. Expected YYYY-MM (e.g. 2026-08).")

    cache_key = (target_month, budget_val, mv_val)
    if not force_refresh and cache_key in _monthly_returns_cache:
        return _monthly_returns_cache[cache_key]

    from models.betting_engine import analyze_race_betting

    strategy_meta_map = {
        "auto": {
            "id": "auto",
            "name": "AI Auto (Optimal)",
            "icon": "🧠",
            "tagline": "AI Dynamic Per-Race Optimal Routing",
        },
        "pure_win": {
            "id": "pure_win",
            "name": "Pure Win (Max ROI)",
            "icon": "⚡",
            "tagline": "100% Single Top Value Pick",
        },
        "hybrid": {
            "id": "hybrid",
            "name": "Balanced (Win + Exotics)",
            "icon": "🎯",
            "tagline": "75% Core Win + 25% Elite Exotics",
        },
        "dutching": {
            "id": "dutching",
            "name": "Dual Dutching (Low DD)",
            "icon": "🛡️",
            "tagline": "Dual Value Win Split — Lowest Drawdown",
        },
    }

    modes = ["auto", "pure_win", "hybrid", "dutching"]

    with get_session() as s:
        # 1. Discover available months with predictions in the database
        month_rows = s.execute(text("""
            SELECT DISTINCT substr(date, 1, 7) AS ym
            FROM races
            WHERE id IN (SELECT race_id FROM predictions)
            ORDER BY ym DESC
        """)).fetchall()
        available_months = [r[0] for r in month_rows]

        if not available_months:
            raise HTTPException(404, "No prediction months available in the database.")

        if target_month not in available_months:
            raise HTTPException(404, f"No predictions found for month {target_month}. Available months: {', '.join(available_months)}")

        # 2. Get distinct race dates for target month
        date_rows = s.execute(
            text("SELECT DISTINCT date FROM races WHERE date LIKE :m ORDER BY date"),
            {"m": f"{target_month}%"},
        ).fetchall()
        race_dates = [r[0] for r in date_rows]

        if not race_dates:
            raise HTTPException(404, f"No race days found for month {target_month}")

        # 3. Process daily returns across race dates in chronological order
        month_stats = {
            m: {
                "total_staked": 0,
                "total_payout": 0,
                "bets_placed": 0,
                "bets_won": 0,
                "pending_staked": 0,
                "pending_races": 0,
            }
            for m in modes
        }

        daily_timeline = []
        total_races_in_month = 0
        total_finished_races_in_month = 0

        for d_str in race_dates:
            try:
                d_res = get_daily_returns(
                    date_str=d_str,
                    budget_per_race=budget_val,
                    model_version=mv_val,
                    force_refresh=force_refresh,
                )
            except Exception as ex:
                log.warning(f"Error computing daily returns for {d_str}: {ex}")
                continue

            d_strats = d_res.get("strategies", {})
            d_total_races = d_res.get("total_races", 0)
            d_has_results = d_res.get("has_results", False)

            total_races_in_month += d_total_races
            if d_has_results:
                total_finished_races_in_month += d_total_races

            d_mode_stats = {}
            for m in modes:
                s_stat = d_strats.get(m, {})
                stk = s_stat.get("staked", 0)
                pay = s_stat.get("payout", 0)
                profit = s_stat.get("profit", 0)
                roi_pct = s_stat.get("roi_pct", 0.0)
                bets_placed = s_stat.get("bets_placed", 0)
                bets_won = s_stat.get("bets_won", 0)

                d_mode_stats[m] = {
                    "staked": stk,
                    "payout": pay,
                    "profit": profit,
                    "roi_pct": roi_pct,
                    "bets_placed": bets_placed,
                    "bets_won": bets_won,
                }

                month_stats[m]["total_staked"] += stk
                month_stats[m]["total_payout"] += pay
                month_stats[m]["bets_placed"] += bets_placed
                month_stats[m]["bets_won"] += bets_won
                month_stats[m]["pending_staked"] += s_stat.get("pending_staked", 0)
                month_stats[m]["pending_races"] += s_stat.get("pending_races", 0)

            auto_d = d_mode_stats.get("auto", {})
            daily_timeline.append({
                "date": d_str,
                "total_races": d_total_races,
                "finished_races": d_total_races if d_has_results else 0,
                "has_results": d_has_results,
                "staked": auto_d.get("staked", 0),
                "payout": auto_d.get("payout", 0),
                "profit": auto_d.get("profit", 0),
                "roi_pct": auto_d.get("roi_pct", 0.0),
                "bets_placed": auto_d.get("bets_placed", 0),
                "bets_won": auto_d.get("bets_won", 0),
                "strategies": d_mode_stats,
            })

        strategies_summary = {}
        for mode in modes:
            stats = month_stats[mode]
            total_staked = stats["total_staked"]
            total_payout = stats["total_payout"]
            bets_placed = stats["bets_placed"]
            bets_won = stats["bets_won"]
            profit = total_payout - total_staked
            roi_pct = round((profit / total_staked * 100), 1) if total_staked > 0 else 0.0
            strike_rate = round((bets_won / bets_placed * 100), 1) if bets_placed > 0 else 0.0
            meta = strategy_meta_map.get(mode, {})

            strategies_summary[mode] = {
                "id": mode,
                "name": meta.get("name", mode.title()),
                "icon": meta.get("icon", "📊"),
                "tagline": meta.get("tagline", ""),
                "staked": total_staked,
                "payout": round(total_payout),
                "profit": round(profit),
                "roi_pct": roi_pct,
                "bets_placed": bets_placed,
                "bets_won": bets_won,
                "strike_rate": strike_rate,
                "pending_staked": stats["pending_staked"],
                "pending_races": stats["pending_races"],
            }

        try:
            m_dt = datetime.strptime(target_month, "%Y-%m")
            month_name = m_dt.strftime("%B %Y")
        except Exception:
            month_name = target_month

        res_obj = {
            "month": target_month,
            "month_name": month_name,
            "has_results": total_finished_races_in_month > 0,
            "total_race_days": len(race_dates),
            "total_races": total_races_in_month,
            "total_finished_races": total_finished_races_in_month,
            "budget_per_race": budget_val,
            "strategies": strategies_summary,
            "daily_timeline": daily_timeline,
            "available_months": available_months,
        }
        _monthly_returns_cache[cache_key] = res_obj
        return res_obj


@app.get("/api/races/all-time-returns")
@app.get("/api/portfolio/all-time")
def get_all_time_returns(
    budget_per_race: int = Query(1000, ge=100, le=1000000, description="Base stake budget per race in yen"),
    model_version: Optional[str] = Query(None, description="Model version for predictions"),
    start_month: Optional[str] = Query(None, description="Starting month in YYYY-MM format"),
    end_month: Optional[str] = Query(None, description="Ending month in YYYY-MM format"),
    force_refresh: bool = False,
):
    """
    Computes all-time aggregated betting performance & returns across all 4 strategy modes
    ('auto', 'pure_win', 'hybrid', 'dutching') from January 2026 to the present/latest month.
    Includes chronological monthly breakdown and high-level portfolio milestones.
    """
    budget_val = _clean_param(budget_per_race) or 1000
    mv_val = _clean_param(model_version)
    sm_val = _clean_param(start_month)
    em_val = _clean_param(end_month)

    cache_key = (budget_val, mv_val, sm_val, em_val)
    if not force_refresh and cache_key in _all_time_returns_cache:
        return _all_time_returns_cache[cache_key]

    modes = ["auto", "pure_win", "hybrid", "dutching"]
    strategy_meta_map = {
        "auto": {
            "id": "auto",
            "name": "AI Auto (Optimal)",
            "icon": "🧠",
            "tagline": "AI Dynamic Per-Race Optimal Routing",
        },
        "pure_win": {
            "id": "pure_win",
            "name": "Pure Win (Max ROI)",
            "icon": "⚡",
            "tagline": "100% Single Top Value Pick",
        },
        "hybrid": {
            "id": "hybrid",
            "name": "Balanced (Win + Exotics)",
            "icon": "🎯",
            "tagline": "75% Core Win + 25% Elite Exotics",
        },
        "dutching": {
            "id": "dutching",
            "name": "Dual Dutching (Low DD)",
            "icon": "🛡️",
            "tagline": "Dual Value Win Split — Lowest Drawdown",
        },
    }

    with get_session() as s:
        # 1. Discover all available prediction months in chronological order
        month_rows = s.execute(text("""
            SELECT DISTINCT substr(date, 1, 7) AS ym
            FROM races
            WHERE id IN (SELECT race_id FROM predictions)
            ORDER BY ym ASC
        """)).fetchall()
        all_available_months = [r[0] for r in month_rows]

        if not all_available_months:
            raise HTTPException(404, "No prediction months available in the database.")

        target_months = [
            m for m in all_available_months
            if (not sm_val or m >= sm_val) and (not em_val or m <= em_val)
        ]

        if not target_months:
            raise HTTPException(404, f"No predictions found in range {sm_val or 'start'} to {em_val or 'end'}. Available months: {', '.join(all_available_months)}")

        # Query earliest and latest dates
        date_bounds = s.execute(text("""
            SELECT min(r.date), max(r.date)
            FROM races r
            WHERE r.id IN (SELECT race_id FROM predictions)
              AND substr(r.date, 1, 7) >= :sm
              AND substr(r.date, 1, 7) <= :em
        """), {"sm": target_months[0], "em": target_months[-1]}).fetchone()
        period_start = date_bounds[0] if date_bounds and date_bounds[0] else f"{target_months[0]}-01"
        period_end = date_bounds[1] if date_bounds and date_bounds[1] else f"{target_months[-1]}-28"

        all_time_stats = {
            m: {
                "total_staked": 0,
                "total_payout": 0,
                "bets_placed": 0,
                "bets_won": 0,
                "pending_staked": 0,
                "pending_races": 0,
            }
            for m in modes
        }

        monthly_breakdown = []
        total_race_days = 0
        total_races = 0
        total_finished_races = 0

        for m_str in target_months:
            try:
                m_data = get_monthly_returns(
                    month_str=m_str,
                    budget_per_race=budget_val,
                    model_version=mv_val,
                    force_refresh=force_refresh,
                )
            except Exception as ex:
                log.warning(f"Error computing monthly returns for {m_str}: {ex}")
                continue

            if not m_data or "strategies" not in m_data:
                continue

            total_race_days += m_data.get("total_race_days", 0)
            total_races += m_data.get("total_races", 0)
            total_finished_races += m_data.get("total_finished_races", 0)

            for mode in modes:
                s_stat = m_data["strategies"].get(mode, {})
                all_time_stats[mode]["total_staked"] += s_stat.get("staked", 0)
                all_time_stats[mode]["total_payout"] += s_stat.get("payout", 0)
                all_time_stats[mode]["bets_placed"] += s_stat.get("bets_placed", 0)
                all_time_stats[mode]["bets_won"] += s_stat.get("bets_won", 0)
                all_time_stats[mode]["pending_staked"] += s_stat.get("pending_staked", 0)
                all_time_stats[mode]["pending_races"] += s_stat.get("pending_races", 0)

            auto_m = m_data["strategies"].get("auto", {})
            monthly_breakdown.append({
                "month": m_str,
                "month_name": m_data.get("month_name", m_str),
                "total_race_days": m_data.get("total_race_days", 0),
                "total_races": m_data.get("total_races", 0),
                "total_finished_races": m_data.get("total_finished_races", 0),
                "has_results": m_data.get("has_results", False),
                "staked": auto_m.get("staked", 0),
                "payout": auto_m.get("payout", 0),
                "profit": auto_m.get("profit", 0),
                "roi_pct": auto_m.get("roi_pct", 0.0),
                "bets_placed": auto_m.get("bets_placed", 0),
                "bets_won": auto_m.get("bets_won", 0),
                "strike_rate": auto_m.get("strike_rate", 0.0),
                "strategies": m_data["strategies"],
            })

        strategies_summary = {}
        for mode in modes:
            stats = all_time_stats[mode]
            staked = stats["total_staked"]
            payout = stats["total_payout"]
            bets_placed = stats["bets_placed"]
            bets_won = stats["bets_won"]
            profit = payout - staked
            roi_pct = round((profit / staked * 100), 1) if staked > 0 else 0.0
            strike_rate = round((bets_won / bets_placed * 100), 1) if bets_placed > 0 else 0.0
            meta = strategy_meta_map.get(mode, {})

            strategies_summary[mode] = {
                "id": mode,
                "name": meta.get("name", mode.title()),
                "icon": meta.get("icon", "📊"),
                "tagline": meta.get("tagline", ""),
                "staked": staked,
                "payout": round(payout),
                "profit": round(profit),
                "roi_pct": roi_pct,
                "bets_placed": bets_placed,
                "bets_won": bets_won,
                "strike_rate": strike_rate,
                "pending_staked": stats["pending_staked"],
                "pending_races": stats["pending_races"],
            }

        # Best month calculation (highest net profit for auto strategy)
        best_month = None
        if monthly_breakdown:
            profitable_months = [m for m in monthly_breakdown if m["profit"] > 0]
            if profitable_months:
                best_month = max(profitable_months, key=lambda x: x["profit"])
            else:
                best_month = max(monthly_breakdown, key=lambda x: x["profit"])

        try:
            start_dt = datetime.strptime(target_months[0], "%Y-%m")
            end_dt = datetime.strptime(target_months[-1], "%Y-%m")
            start_str = start_dt.strftime("%B %Y")
            end_str = end_dt.strftime("%B %Y")
            period_label = f"{start_str} – Present" if target_months[-1] == all_available_months[-1] else f"{start_str} – {end_str}"
        except Exception:
            period_label = f"{target_months[0]} – {target_months[-1]}"

        res_obj = {
            "period_start": period_start,
            "period_end": period_end,
            "period_label": period_label,
            "start_month": target_months[0],
            "end_month": target_months[-1],
            "has_results": total_finished_races > 0,
            "total_race_days": total_race_days,
            "total_races": total_races,
            "total_finished_races": total_finished_races,
            "budget_per_race": budget_val,
            "strategies": strategies_summary,
            "monthly_breakdown": monthly_breakdown,
            "best_month": best_month,
            "available_months": list(reversed(all_available_months)),
        }
        _all_time_returns_cache[cache_key] = res_obj
        return res_obj


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
    max_odds: float = Query(60.0, description="Max odds for value bets"),
    min_odds: float = Query(2.0, description="Min odds for value bets"),
    refresh_odds: bool = Query(True, description="Whether to fetch latest live odds first"),
):
    """
    Run model predictions for ALL races on a given date (YYYY-MM-DD),
    persisting predictions and value bets into the database.
    """
    model = version if (isinstance(version, str) and version) else MODEL_VERSION
    ev_thresh = float(ev_threshold) if isinstance(ev_threshold, (int, float)) else 0.08
    max_o = float(max_odds) if isinstance(max_odds, (int, float)) else 60.0
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

    # 2. Optionally refresh live odds (concurrently outside DB transaction)
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

        def _fetch_race_odds(r_id: int, nk_id: str):
            if not nk_id:
                return (r_id, None)
            try:
                odds = fetch_win_odds(nk_id)
                return (r_id, odds)
            except Exception:
                return (r_id, None)

        odds_results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(_fetch_race_odds, r.id, r.netkeiba_id)
                for r in race_rows
                if r.netkeiba_id
            ]
            for f in as_completed(futures):
                try:
                    r_id, odds = f.result()
                    if odds:
                        official_dt = odds[0].get("official_datetime") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        odds_results.append((r_id, official_dt, odds))
                except Exception:
                    pass

        if odds_results:
            with get_session() as s:
                for r_id, official_dt, odds in odds_results:
                    s.execute(
                        text("UPDATE races SET odds_updated_at = :dt WHERE id = :race_id"),
                        {"dt": official_dt, "race_id": r_id},
                    )
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
                                "race_id": r_id,
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

    # 4. Persist predictions + value bets into DB with fast bulk insert
    now = datetime.utcnow().isoformat()
    value_bets_count = 0
    races_with_value_bets = set()

    pred_params = []
    vb_params = []
    for _, row in df.iterrows():
        rid = int(row["race_id"])
        eid = int(row["entry_id"])
        ev_val = float(row["ev"]) if (row["ev"] is not None and not pd.isna(row["ev"])) else None
        pred_params.append({
            "race_id": rid,
            "entry_id": eid,
            "mv": model,
            "win_prob": float(row["combined_prob"]),
            "edge": ev_val,
            "created_at": now,
        })
        if row.get("is_value_bet") and ev_val is not None:
            value_bets_count += 1
            races_with_value_bets.add(rid)
            vb_params.append({
                "race_id": rid,
                "entry_id": eid,
                "model_prob": float(row["combined_prob"]),
                "market_prob": float(row["market_prob"]) if row["market_prob"] is not None else 0.0,
                "ev": ev_val,
                "kelly": float(row["kelly_fraction"]),
                "stake": int(row["recommended_stake"]),
                "mv": model,
                "created_at": now,
            })

    with get_session() as s:
        id_list_str = ",".join(str(rid) for rid in race_ids)
        s.execute(text(f"DELETE FROM predictions WHERE race_id IN ({id_list_str}) AND model_version = :mv"), {"mv": model})
        s.execute(text(f"DELETE FROM value_bets WHERE race_id IN ({id_list_str}) AND model_version = :mv"), {"mv": model})

        if pred_params:
            s.execute(
                text("""
                    INSERT INTO predictions
                        (race_id, entry_id, model_version, win_prob, edge, created_at)
                    VALUES
                        (:race_id, :entry_id, :mv, :win_prob, :edge, :created_at)
                """),
                pred_params,
            )
        if vb_params:
            s.execute(
                text("""
                    INSERT INTO value_bets
                        (race_id, entry_id, bet_type, model_prob, market_prob,
                         ev, kelly_fraction, recommended_stake, model_version, created_at)
                    VALUES
                        (:race_id, :entry_id, 'win', :model_prob, :market_prob,
                         :ev, :kelly, :stake, :mv, :created_at)
                """),
                vb_params,
            )
        s.commit()

    _daily_returns_cache.clear()
    _monthly_returns_cache.clear()

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
    _daily_returns_cache.clear()
    _monthly_returns_cache.clear()
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

    # 2. Fetch new odds outside DB transaction
    from scraper.odds_watcher import fetch_win_odds
    odds = fetch_win_odds(nk_id) if nk_id else []
    if odds:
        official_dt = odds[0].get("official_datetime") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_session() as s:
            s.execute(
                text("UPDATE races SET odds_updated_at = :dt WHERE id = :race_id"),
                {"dt": official_dt, "race_id": race_id},
            )
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
            max_odds=60.0,
            min_odds=2.0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="No predictions generated")

    # 4. Persist predictions + value bets to the DB with bulk insert
    now = datetime.utcnow().isoformat()
    pred_params = []
    vb_params = []
    for _, row in df.iterrows():
        ev_val = float(row["ev"]) if (row["ev"] is not None and not pd.isna(row["ev"])) else None
        pred_params.append({
            "race_id": int(row["race_id"]),
            "entry_id": int(row["entry_id"]),
            "mv": model,
            "win_prob": float(row["combined_prob"]),
            "edge": ev_val,
            "created_at": now,
        })
        if row.get("is_value_bet") and ev_val is not None:
            vb_params.append({
                "race_id": int(row["race_id"]),
                "entry_id": int(row["entry_id"]),
                "model_prob": float(row["combined_prob"]),
                "market_prob": float(row["market_prob"]) if row["market_prob"] is not None else 0.0,
                "ev": ev_val,
                "kelly": float(row["kelly_fraction"]),
                "stake": int(row["recommended_stake"]),
                "mv": model,
                "created_at": now,
            })

    with get_session() as s:
        s.execute(
            text("DELETE FROM predictions WHERE race_id = :race_id AND model_version = :mv"),
            {"race_id": race_id, "mv": model},
        )
        s.execute(
            text("DELETE FROM value_bets WHERE race_id = :race_id AND model_version = :mv"),
            {"race_id": race_id, "mv": model},
        )
        if pred_params:
            s.execute(
                text("""
                    INSERT INTO predictions
                        (race_id, entry_id, model_version, win_prob, edge, created_at)
                    VALUES
                        (:race_id, :entry_id, :mv, :win_prob, :edge, :created_at)
                """),
                pred_params,
            )
        if vb_params:
            s.execute(
                text("""
                    INSERT INTO value_bets
                        (race_id, entry_id, bet_type, model_prob, market_prob,
                         ev, kelly_fraction, recommended_stake, model_version, created_at)
                    VALUES
                        (:race_id, :entry_id, 'win', :model_prob, :market_prob,
                         :ev, :kelly, :stake, :mv, :created_at)
                """),
                vb_params,
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
        _daily_returns_cache.clear()
        _monthly_returns_cache.clear()
        _all_time_returns_cache.clear()
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

