"""
UmaEdge — FastAPI Backend API.

Serves race data, model predictions, agent analysis, bankroll metrics,
exotic ticket suggestions, backtest results, and odds time-series
from the horsebet schema in Supabase.
"""

import os
from datetime import date, datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client

load_dotenv()

# ---------------------------------------------------------------------------
# Supabase Client
# ---------------------------------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="UmaEdge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: query the horsebet schema
# ---------------------------------------------------------------------------

def table(name: str):
    """Shortcut to query horsebet.<name> via Supabase."""
    return supabase.schema("horsebet").table(name)


# ===================================================================
# 1. RACES
# ===================================================================

@app.get("/api/races")
def list_races(
    upcoming: bool = Query(False, description="If true, only future races"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    course: Optional[str] = None,
    grade: Optional[str] = None,
):
    """List races with optional filters."""
    q = table("races").select(
        "*, courses(name, name_jp, surface)"
    ).order("date", desc=True).range(offset, offset + limit - 1)

    if upcoming:
        q = q.gte("date", date.today().isoformat())
    if course:
        q = q.eq("courses.name", course)
    if grade:
        q = q.eq("grade", grade)

    res = q.execute()
    return {"races": res.data, "count": len(res.data)}


@app.get("/api/races/{race_id}")
def get_race(race_id: int):
    """Full race detail: race info + entries + results + predictions."""
    race = table("races").select(
        "*, courses(name, name_jp, surface, direction)"
    ).eq("id", race_id).maybe_single().execute()

    if not race.data:
        raise HTTPException(404, f"Race {race_id} not found")

    entries = table("entries").select(
        "*, horses(name, name_jp, sex, birth_year), "
        "jockeys(name, name_jp), "
        "results(finish_pos, margin, time_secs, last_3f_secs, corner_positions, running_style)"
    ).eq("race_id", race_id).order("post_position").execute()

    predictions = table("predictions").select("*").eq(
        "race_id", race_id
    ).execute()

    value_bets = table("value_bets").select("*").eq(
        "race_id", race_id
    ).execute()

    return {
        "race": race.data,
        "entries": entries.data,
        "predictions": predictions.data,
        "value_bets": value_bets.data,
    }


# ===================================================================
# 2. PREDICTIONS
# ===================================================================

@app.get("/api/predictions/{race_id}")
def get_predictions(race_id: int, model_version: str = "latest"):
    """Model predictions + value bets for a race."""
    pq = table("predictions").select(
        "*, entries(post_position, draw, horses(name, name_jp), jockeys(name, name_jp))"
    ).eq("race_id", race_id)

    if model_version != "latest":
        pq = pq.eq("model_version", model_version)

    predictions = pq.order("win_prob", desc=True).execute()

    value_bets = table("value_bets").select("*").eq(
        "race_id", race_id
    ).order("ev", desc=True).execute()

    return {
        "predictions": predictions.data,
        "value_bets": value_bets.data,
    }


# ===================================================================
# 3. AI ANALYSIS
# ===================================================================

@app.get("/api/analysis/{race_id}")
def get_analysis(race_id: int, language: str = "en"):
    """Get cached LLM race analysis or return empty."""
    res = table("agent_analyses").select("*").eq(
        "race_id", race_id
    ).eq("language", language).order(
        "created_at", desc=True
    ).limit(1).execute()

    if not res.data:
        return {"analysis": None, "message": "No analysis available for this race."}

    return {"analysis": res.data[0]}


# ===================================================================
# 4. BANKROLL
# ===================================================================

@app.get("/api/bankroll")
def get_bankroll():
    """Current bankroll summary: balance, metrics, recent P&L."""
    # Latest entry for running balance
    latest = table("bankroll_log").select(
        "running_balance, date, created_at"
    ).order("created_at", desc=True).limit(1).execute()

    balance = latest.data[0]["running_balance"] if latest.data else 100_000

    # Last 90 days for chart
    history = table("bankroll_log").select(
        "date, running_balance, profit, stake, payout"
    ).order("created_at").execute()

    # Today's P&L
    today_str = date.today().isoformat()
    today_bets = table("bankroll_log").select(
        "stake, payout, profit"
    ).eq("date", today_str).execute()

    today_pnl = sum(row.get("profit", 0) or 0 for row in today_bets.data)
    today_staked = sum(row.get("stake", 0) or 0 for row in today_bets.data)

    # Peak & drawdown
    balances = [r["running_balance"] for r in history.data if r.get("running_balance")]
    peak = max(balances) if balances else balance
    drawdown = (peak - balance) / peak * 100 if peak > 0 else 0

    # Win stats
    all_bets = history.data
    total_bets = len(all_bets)
    winning = sum(1 for b in all_bets if (b.get("profit") or 0) > 0)
    total_staked = sum(b.get("stake", 0) or 0 for b in all_bets)
    total_profit = sum(b.get("profit", 0) or 0 for b in all_bets)
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    return {
        "balance": balance,
        "today_pnl": today_pnl,
        "today_staked": today_staked,
        "peak_balance": peak,
        "drawdown_pct": round(drawdown, 2),
        "total_bets": total_bets,
        "winning_bets": winning,
        "win_rate": round(winning / total_bets * 100, 1) if total_bets else 0,
        "roi_pct": round(roi, 2),
        "total_staked": total_staked,
        "total_profit": total_profit,
        "history": history.data,
    }


@app.get("/api/bankroll/log")
def get_bankroll_log(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Paginated bet log."""
    res = table("bankroll_log").select(
        "*, races(race_name, race_name_jp, date, courses(name))"
    ).order("created_at", desc=True).range(offset, offset + limit - 1).execute()

    return {"log": res.data, "count": len(res.data)}


# ===================================================================
# 5. EXOTIC TICKETS
# ===================================================================

@app.get("/api/exotic/{race_id}")
def get_exotic_tickets(race_id: int, bet_type: Optional[str] = None):
    """Optimal exotic bet tickets for a race."""
    q = table("exotic_tickets").select("*").eq("race_id", race_id)

    if bet_type:
        q = q.eq("bet_type", bet_type)

    res = q.order("created_at", desc=True).execute()
    return {"tickets": res.data}


# ===================================================================
# 6. BACKTEST
# ===================================================================

@app.get("/api/backtest")
def get_backtest(
    model_version: str = Query("latest"),
    grade: Optional[str] = None,
    course: Optional[str] = None,
):
    """
    Backtest summary: aggregate metrics + P&L curve.
    Reads from predictions + results joined data.
    """
    # Query predictions with results for backtesting
    q = table("predictions").select(
        "race_id, entry_id, win_prob, model_version, "
        "entries(post_position, odds_win, horse_id, "
        "  results(finish_pos), "
        "  horses(name, name_jp)), "
        "races(date, race_name, grade, course_id, courses(name))"
    )

    if model_version != "latest":
        q = q.eq("model_version", model_version)

    res = q.order("race_id").execute()
    data = res.data or []

    # Compute backtest metrics locally
    bets = []
    for pred in data:
        entry = pred.get("entries") or {}
        results = entry.get("results")
        result = results[0] if isinstance(results, list) and results else results
        odds = entry.get("odds_win") or 0
        if odds <= 1:
            continue

        model_prob = pred["win_prob"]
        market_prob = 1 / odds if odds > 0 else 1
        ev = model_prob / market_prob - 1 if market_prob > 0 else 0

        if ev > 0.05:  # EV threshold
            finish = result.get("finish_pos") if result else None
            won = finish == 1 if finish else False
            payout = 1000 * odds if won else 0
            profit = payout - 1000

            race_info = pred.get("races") or {}
            bets.append({
                "race_id": pred["race_id"],
                "date": race_info.get("date"),
                "race_name": race_info.get("race_name"),
                "grade": race_info.get("grade"),
                "course": (race_info.get("courses") or {}).get("name"),
                "horse": (entry.get("horses") or {}).get("name"),
                "model_prob": round(model_prob, 4),
                "market_prob": round(market_prob, 4),
                "ev": round(ev, 4),
                "odds": odds,
                "won": won,
                "stake": 1000,
                "payout": payout,
                "profit": profit,
            })

    # Apply optional filters
    if grade:
        bets = [b for b in bets if b.get("grade") == grade]
    if course:
        bets = [b for b in bets if b.get("course") == course]

    # Aggregate metrics
    total_bets = len(bets)
    wins = sum(1 for b in bets if b["won"])
    total_staked = sum(b["stake"] for b in bets)
    total_payout = sum(b["payout"] for b in bets)
    total_profit = total_payout - total_staked

    # Build P&L curve
    balance = 100_000
    curve = []
    for b in bets:
        balance += b["profit"]
        curve.append({"date": b["date"], "balance": balance, "profit": b["profit"]})

    # ROI by category
    roi_by_grade = {}
    for b in bets:
        g = b.get("grade") or "Other"
        if g not in roi_by_grade:
            roi_by_grade[g] = {"staked": 0, "profit": 0, "count": 0}
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
        "total_profit": total_profit,
        "roi_pct": round(total_profit / total_staked * 100, 2) if total_staked else 0,
        "balance_curve": curve,
        "roi_by_grade": roi_by_grade,
        "bets": bets[:50],  # limit response size
    }


# ===================================================================
# 7. ODDS
# ===================================================================

@app.get("/api/odds/{race_id}")
def get_odds(race_id: int, bet_type: str = "win"):
    """Odds time-series for a race."""
    res = table("odds_snapshots").select("*").eq(
        "race_id", race_id
    ).eq("bet_type", bet_type).order("captured_at").execute()

    return {"odds": res.data}


# ===================================================================
# Health
# ===================================================================

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "umaedge-api", "timestamp": datetime.utcnow().isoformat()}
