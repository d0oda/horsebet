"""
UmaEdge — Rigorous Betting Strategy Evaluator & Optimizer.

Evaluates and optimizes multi-ticket betting strategies across historical races
with recorded predictions and actual outcomes in the database.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import text

from scraper.db import get_session
from models.betting_engine import (
    calculate_pricing_breakdown,
    JointFinishModel,
    HorsePricing,
    construct_staking_plan,
)

TAKEOUT_MAP = {
    "win": 0.20,
    "place": 0.20,
    "wide": 0.225,
    "quinella": 0.225,
    "exacta": 0.25,
    "trio": 0.25,
    "trifecta": 0.275,
}


@dataclass
class StrategyMetrics:
    name: str
    total_races: int = 0
    races_bet: int = 0
    total_tickets: int = 0
    winning_tickets: int = 0
    total_staked: int = 0
    total_payout: int = 0
    net_profit: int = 0
    roi_pct: float = 0.0
    hit_rate: float = 0.0
    race_hit_rate: float = 0.0
    max_drawdown_yen: int = 0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    avg_odds_bet: float = 0.0
    avg_ev_bet: float = 0.0


def load_historical_eval_data(model_version: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load historical races with predictions, odds, and verified finish positions."""
    q = """
        SELECT
            r.id as race_id,
            r.date,
            r.race_number,
            c.name as venue,
            r.race_name_jp as race_name,
            e.id as entry_id,
            e.post_position,
            e.odds_win,
            e.popularity,
            e.finish_pos,
            h.name_jp as horse_name_jp,
            h.name as horse_name,
            p.win_prob,
            p.model_version
        FROM predictions p
        JOIN entries e ON e.id = p.entry_id
        JOIN races r ON r.id = p.race_id
        LEFT JOIN courses c ON c.id = r.course_id
        LEFT JOIN horses h ON h.id = e.horse_id
        WHERE e.finish_pos IS NOT NULL AND e.odds_win IS NOT NULL
    """
    params = {}
    if model_version:
        q += " AND p.model_version = :mv"
        params["mv"] = model_version
    else:
        # Pick latest model or retrain_20260507_1645
        q += """
            AND p.model_version = 'retrain_20260507_1645'
        """
    q += " ORDER BY r.date, r.id, e.post_position"

    with get_session() as session:
        rows = session.execute(text(q), params).fetchall()

    if not rows:
        # If specific version returned none, load all with finish_pos
        with get_session() as session:
            rows = session.execute(text("""
                SELECT
                    r.id as race_id,
                    r.date,
                    r.race_number,
                    c.name as venue,
                    r.race_name_jp as race_name,
                    e.id as entry_id,
                    e.post_position,
                    e.odds_win,
                    e.popularity,
                    e.finish_pos,
                    h.name_jp as horse_name_jp,
                    h.name as horse_name,
                    p.win_prob,
                    p.model_version
                FROM predictions p
                JOIN entries e ON e.id = p.entry_id
                JOIN races r ON r.id = p.race_id
                LEFT JOIN courses c ON c.id = r.course_id
                LEFT JOIN horses h ON h.id = e.horse_id
                WHERE e.finish_pos IS NOT NULL AND e.odds_win IS NOT NULL
                ORDER BY r.date, r.id, e.post_position
            """)).fetchall()

    # Group by race_id
    races_dict = {}
    for row in rows:
        rid = row.race_id
        if rid not in races_dict:
            races_dict[rid] = {
                "race_id": rid,
                "date": str(row.date),
                "race_number": row.race_number,
                "venue": row.venue,
                "race_name": row.race_name,
                "entries": [],
            }
        races_dict[rid]["entries"].append({
            "entry_id": row.entry_id,
            "post_position": row.post_position,
            "horse_name": row.horse_name,
            "horse_name_jp": row.horse_name_jp,
            "odds": float(row.odds_win or 1.0),
            "win_prob": float(row.win_prob or 0.0),
            "finish_pos": int(row.finish_pos) if row.finish_pos is not None else None,
        })

    return list(races_dict.values())


def evaluate_ticket_payout(
    ticket_type_en: str,
    selection: List[int],
    stake: int,
    market_odds: float,
    finish_map: Dict[int, int],
    odds_map: Dict[int, float],
) -> Tuple[bool, int]:
    """
    Determine if a ticket won based on finish positions and compute payout.
    finish_map: {post_position: finish_pos (1-indexed)}
    """
    if stake <= 0:
        return False, 0

    if ticket_type_en == "win":
        pp = selection[0]
        if finish_map.get(pp) == 1:
            win_odds = odds_map.get(pp, market_odds)
            return True, int(math.floor(stake * win_odds))
        return False, 0

    elif ticket_type_en == "place":
        pp = selection[0]
        # In field >= 8, top 3 win place
        pos = finish_map.get(pp)
        if pos is not None and 1 <= pos <= 3:
            # Place payout estimated as 0.30 of win odds
            place_odds = max(1.1, round(odds_map.get(pp, market_odds) * 0.30, 1))
            return True, int(math.floor(stake * place_odds))
        return False, 0

    elif ticket_type_en == "exacta":
        p1, p2 = selection[0], selection[1]
        if finish_map.get(p1) == 1 and finish_map.get(p2) == 2:
            return True, int(math.floor(stake * market_odds))
        return False, 0

    elif ticket_type_en == "quinella":
        p1, p2 = selection[0], selection[1]
        pos1, pos2 = finish_map.get(p1), finish_map.get(p2)
        if pos1 is not None and pos2 is not None and {pos1, pos2} == {1, 2}:
            return True, int(math.floor(stake * market_odds))
        return False, 0

    elif ticket_type_en == "wide":
        p1, p2 = selection[0], selection[1]
        pos1, pos2 = finish_map.get(p1), finish_map.get(p2)
        if pos1 is not None and pos2 is not None and pos1 <= 3 and pos2 <= 3:
            return True, int(math.floor(stake * market_odds))
        return False, 0

    elif ticket_type_en == "trio":
        p1, p2, p3 = selection[0], selection[1], selection[2]
        pos1, pos2, pos3 = finish_map.get(p1), finish_map.get(p2), finish_map.get(p3)
        if pos1 is not None and pos2 is not None and pos3 is not None and {pos1, pos2, pos3} == {1, 2, 3}:
            return True, int(math.floor(stake * market_odds))
        return False, 0

    return False, 0


def backtest_custom_strategy(
    races: List[Dict[str, Any]],
    name: str,
    min_ev: float = 0.05,
    min_win_prob: float = 0.04,
    min_odds: float = 1.5,
    max_odds: float = 40.0,
    strategy_mode: str = "dynamic_portfolio", # "win_only", "dynamic_portfolio", "kelly_win", "exotic_blend"
    budget: int = 1000,
    kelly_fraction: float = 0.25,
) -> StrategyMetrics:
    """Run backtest for a specific strategy configuration across historical races."""
    total_staked = 0
    total_payout = 0
    total_tickets = 0
    winning_tickets = 0
    races_bet = 0
    winning_races = 0

    pnl_history = []
    daily_pnl = {}
    odds_bet_list = []
    ev_bet_list = []

    running_bankroll = 100_000
    peak_bankroll = 100_000
    max_drawdown_yen = 0
    max_drawdown_pct = 0.0

    for r in races:
        entries = r["entries"]
        if len(entries) < 3:
            continue

        finish_map = {e["post_position"]: e["finish_pos"] for e in entries if e["finish_pos"] is not None}
        odds_map = {e["post_position"]: e["odds"] for e in entries}

        # Calculate pricing
        pricing = calculate_pricing_breakdown(entries)

        # Apply filtering constraints
        # Candidate value runners:
        eligible_pricing = [
            h for h in pricing
            if h.ev >= min_ev
            and h.win_prob >= min_win_prob
            and min_odds <= h.market_odds <= max_odds
        ]

        if not eligible_pricing:
            continue

        # Strategy Execution
        race_staked = 0
        race_payout = 0
        race_won = False

        if strategy_mode == "win_only_flat":
            anchor = eligible_pricing[0]
            stake = budget
            won, payout = evaluate_ticket_payout(
                "win", [anchor.post_position], stake, anchor.market_odds, finish_map, odds_map
            )
            total_tickets += 1
            if won:
                winning_tickets += 1
                race_won = True
            race_staked += stake
            race_payout += payout
            odds_bet_list.append(anchor.market_odds)
            ev_bet_list.append(anchor.ev)

        elif strategy_mode == "kelly_win":
            anchor = eligible_pricing[0]
            b = anchor.market_odds - 1.0
            p = anchor.win_prob
            q = 1.0 - p
            full_kelly = max(0.0, (b * p - q) / b) if b > 0 else 0.0
            k_stake = int(math.floor((running_bankroll * full_kelly * kelly_fraction) / 100.0) * 100)
            stake = min(max(100, k_stake), int(running_bankroll * 0.05)) # Cap at 5% bankroll

            won, payout = evaluate_ticket_payout(
                "win", [anchor.post_position], stake, anchor.market_odds, finish_map, odds_map
            )
            total_tickets += 1
            if won:
                winning_tickets += 1
                race_won = True
            race_staked += stake
            race_payout += payout
            odds_bet_list.append(anchor.market_odds)
            ev_bet_list.append(anchor.ev)

        elif strategy_mode == "dynamic_portfolio":
            plan = construct_staking_plan(pricing, budget=budget)
            # Filter plan tickets by min_ev
            viable_tickets = [t for t in plan.portfolio_tickets if (t.ev or 0.0) >= min_ev]
            if not viable_tickets:
                continue

            for t in viable_tickets:
                won, payout = evaluate_ticket_payout(
                    t.ticket_type_en, t.selection, t.stake, t.market_odds or t.fair_odds, finish_map, odds_map
                )
                total_tickets += 1
                if won:
                    winning_tickets += 1
                    race_won = True
                race_staked += t.stake
                race_payout += payout
                if t.market_odds:
                    odds_bet_list.append(t.market_odds)
                if t.ev is not None:
                    ev_bet_list.append(t.ev)

        if race_staked > 0:
            races_bet += 1
            if race_won:
                winning_races += 1
            total_staked += race_staked
            total_payout += race_payout
            profit = race_payout - race_staked
            pnl_history.append(profit)

            # Bankroll tracking
            running_bankroll += profit
            if running_bankroll > peak_bankroll:
                peak_bankroll = running_bankroll
            dd = peak_bankroll - running_bankroll
            dd_pct = dd / peak_bankroll if peak_bankroll > 0 else 0.0
            if dd > max_drawdown_yen:
                max_drawdown_yen = dd
            if dd_pct > max_drawdown_pct:
                max_drawdown_pct = dd_pct

            # Daily aggregation
            date_str = r["date"]
            daily_pnl[date_str] = daily_pnl.get(date_str, 0) + profit

    net_profit = total_payout - total_staked
    roi_pct = (net_profit / total_staked * 100.0) if total_staked > 0 else 0.0
    hit_rate = (winning_tickets / total_tickets * 100.0) if total_tickets > 0 else 0.0
    race_hit_rate = (winning_races / races_bet * 100.0) if races_bet > 0 else 0.0

    # Sharpe calculation
    if len(pnl_history) > 1:
        returns = np.array(pnl_history)
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        sharpe = (mean_r / std_r) * np.sqrt(len(pnl_history)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    gross_wins = sum(p for p in pnl_history if p > 0)
    gross_losses = abs(sum(p for p in pnl_history if p < 0))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else 999.0

    return StrategyMetrics(
        name=name,
        total_races=len(races),
        races_bet=races_bet,
        total_tickets=total_tickets,
        winning_tickets=winning_tickets,
        total_staked=total_staked,
        total_payout=total_payout,
        net_profit=net_profit,
        roi_pct=round(roi_pct, 2),
        hit_rate=round(hit_rate, 2),
        race_hit_rate=round(race_hit_rate, 2),
        max_drawdown_yen=int(max_drawdown_yen),
        max_drawdown_pct=round(max_drawdown_pct * 100, 2),
        sharpe=round(sharpe, 3),
        profit_factor=round(profit_factor, 3),
        avg_odds_bet=round(float(np.mean(odds_bet_list)), 2) if odds_bet_list else 0.0,
        avg_ev_bet=round(float(np.mean(ev_bet_list)), 3) if ev_bet_list else 0.0,
    )


def run_full_evaluation():
    print("Loading historical races from database...")
    races = load_historical_eval_data()
    print(f"Loaded {len(races)} historical races with actual outcomes.")

    results: List[StrategyMetrics] = []

    # 1. Baseline: Flat Win Bet (Varying EV thresholds)
    for ev in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        m = backtest_custom_strategy(
            races,
            name=f"Flat Win (EV >= {int(ev*100)}%)",
            min_ev=ev,
            min_win_prob=0.04,
            strategy_mode="win_only_flat",
        )
        results.append(m)

    # 2. Probability Floor Variations on Win Bets
    for p_floor in [0.04, 0.08, 0.12, 0.15, 0.20]:
        m = backtest_custom_strategy(
            races,
            name=f"Flat Win (EV>=15%, Prob>={int(p_floor*100)}%)",
            min_ev=0.15,
            min_win_prob=p_floor,
            strategy_mode="win_only_flat",
        )
        results.append(m)

    # 3. Fractional Kelly Sizing
    for kf in [0.10, 0.20, 0.25, 0.33, 0.50]:
        m = backtest_custom_strategy(
            races,
            name=f"Kelly Win (EV>=15%, Kelly={int(kf*100)}%)",
            min_ev=0.15,
            min_win_prob=0.06,
            strategy_mode="kelly_win",
            kelly_fraction=kf,
        )
        results.append(m)

    # 4. Dynamic Portfolio (Win + Positive EV Exotics)
    for ev in [0.0, 0.05, 0.10, 0.15, 0.20]:
        m = backtest_custom_strategy(
            races,
            name=f"Dynamic Portfolio (EV >= {int(ev*100)}%)",
            min_ev=ev,
            min_win_prob=0.05,
            strategy_mode="dynamic_portfolio",
            budget=1000,
        )
        results.append(m)

    # Format Results Table
    df = pd.DataFrame([
        {
            "Strategy": r.name,
            "Races Bet": f"{r.races_bet}/{r.total_races} ({r.races_bet/r.total_races*100:.1f}%)",
            "Hit Rate": f"{r.hit_rate}%",
            "Total Staked": f"¥{r.total_staked:,}",
            "Total Payout": f"¥{r.total_payout:,}",
            "Net Profit": f"{'+' if r.net_profit > 0 else ''}¥{r.net_profit:,}",
            "ROI": f"{'+' if r.roi_pct > 0 else ''}{r.roi_pct:.1f}%",
            "Max DD %": f"{r.max_drawdown_pct:.1f}%",
            "Sharpe": r.sharpe,
            "Profit Factor": r.profit_factor,
            "Avg Odds": f"{r.avg_odds_bet}x",
            "Avg EV": f"{r.avg_ev_bet:+.1%}",
        }
        for r in results
    ])

    print("\n" + "="*110)
    print("HISTORICAL BETTING STRATEGY EVALUATION & OPTIMIZATION RESULTS")
    print("="*110)
    print(df.to_string(index=False))
    print("="*110)

    # Save to CSV
    df.to_csv("data/betting_strategy_evaluation_results.csv", index=False)
    print("\nSaved full evaluation results to data/betting_strategy_evaluation_results.csv")


if __name__ == "__main__":
    run_full_evaluation()
