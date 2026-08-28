"""
UmaEdge — Full-Database Multi-Strategy Benchmark & Adaptive Meta-Staking Evaluator.

Evaluates all betting styles across historical races and benchmarks the
Adaptive Per-Race Strategy Router against static allocation templates.

Usage:
    .venv/bin/python -m models.benchmark_all_strategies --test-start 2024-01-01
    .venv/bin/python -m models.benchmark_all_strategies --full-dataset
"""

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from models.betting_engine import (
    HorsePricing,
    JointFinishModel,
    calculate_pricing_breakdown,
    construct_staking_plan,
    select_adaptive_strategy,
)
from models.train import (
    cls_to_probs,
    ensemble_predict,
    load_model,
    scores_to_probs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("benchmark")

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
class BenchmarkResult:
    strategy_name: str
    total_races: int = 0
    races_bet: int = 0
    total_tickets: int = 0
    winning_tickets: int = 0
    total_staked: int = 0
    total_payout: int = 0
    net_profit: int = 0
    roi_pct: float = 0.0
    ticket_hit_rate: float = 0.0
    race_win_rate: float = 0.0
    max_drawdown_yen: int = 0
    max_drawdown_pct: float = 0.0
    daily_sharpe_padded: float = 0.0
    daily_sharpe_active: float = 0.0
    per_bet_sharpe: float = 0.0
    profit_factor: float = 0.0
    avg_odds: float = 0.0
    avg_ev: float = 0.0
    mode_counts: Dict[str, int] = field(default_factory=dict)
    mode_profits: Dict[str, int] = field(default_factory=dict)
    mode_stakes: Dict[str, int] = field(default_factory=dict)


def generate_dataset_predictions(
    features_path: str = "data/features.parquet",
    model_version: str = "retrain_20260822_2143",
    test_start_date: Optional[str] = "2024-01-01",
) -> pd.DataFrame:
    """Load features parquet and generate high-fidelity ensemble predictions."""
    log.info(f"Loading features from {features_path}...")
    t0 = time.time()
    df = pd.read_parquet(features_path)
    log.info(f"Loaded {len(df):,} total feature rows in {time.time()-t0:.2f}s")

    df["date"] = df["date"].astype(str)
    
    # Filter valid finishes and odds
    df = df[df["finish_pos"].notna() & df["odds_win"].notna()].copy()
    df = df[df["odds_win"] > 1.0].copy()

    if test_start_date:
        log.info(f"Filtering to out-of-sample test window >= {test_start_date}...")
        df = df[df["date"] >= test_start_date].copy()
    
    log.info(f"Test slice contains {len(df):,} runner entries across {df['race_id'].nunique():,} races.")

    log.info(f"Loading trained model '{model_version}'...")
    lgb_model, xgb_model, meta = load_model(model_version)
    feature_cols = meta["feature_cols"]

    # Align categoricals
    cat_cols = ["sire_id", "broodmare_sire_id", "going_code", "surface_code", "draw"]
    categories_map = meta.get("categories", {})
    for col in cat_cols:
        if col in df.columns:
            val = df[col].fillna("Unknown").astype(str)
            saved_cats = categories_map.get(col)
            if saved_cats:
                df[col] = pd.Categorical(val, categories=saved_cats)
            else:
                df[col] = val.astype("category")

    # Predict classifier
    X = df[feature_cols]
    log.info("Generating LightGBM & XGBoost classifier predictions...")
    lgb_probs = lgb_model.predict(X)
    xgb_probs = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
    ensemble_cls = ensemble_predict(lgb_probs, xgb_probs)

    race_ids = df["race_id"].values
    prob_cls = cls_to_probs(ensemble_cls, race_ids)

    # Regression blend
    lgb_reg_model = meta.get("lgb_reg_model")
    xgb_reg_model = meta.get("xgb_reg_model")
    prob_reg = np.zeros_like(prob_cls)
    if lgb_reg_model is not None and xgb_reg_model is not None:
        log.info("Generating regression predictions...")
        lgb_reg_p = lgb_reg_model.predict(X)
        xgb_reg_p = xgb_reg_model.predict(xgb.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
        ens_reg = ensemble_predict(lgb_reg_p, xgb_reg_p)
        ens_reg = np.maximum(ens_reg, 0.0)
        prob_reg = scores_to_probs(ens_reg, race_ids)

    # Ranker blend
    lgb_rank_model = meta.get("lgb_rank_model")
    xgb_rank_model = meta.get("xgb_rank_model")
    prob_rnk = np.zeros_like(prob_cls)
    if lgb_rank_model is not None and xgb_rank_model is not None:
        log.info("Generating ranker predictions...")
        lgb_rnk_p = lgb_rank_model.predict(X)
        xgb_rnk_p = xgb_rank_model.predict(xgb.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
        ens_rnk = ensemble_predict(lgb_rnk_p, xgb_rnk_p)
        prob_rnk = scores_to_probs(ens_rnk, race_ids)

    # Goldilocks Blend (0.7 Cls + 0.2 Reg + 0.1 Rnk)
    final_probs = 0.7 * prob_cls + 0.2 * prob_reg + 0.1 * prob_rnk

    # Calibration
    calibrator = meta.get("calibrator")
    if calibrator is not None:
        cal_method = meta.get("calibration_method", "isotonic")
        if cal_method == "platt":
            final_probs = calibrator.predict_proba(final_probs.reshape(-1, 1))[:, 1]
        elif cal_method == "isotonic":
            final_probs = calibrator.predict(final_probs)
        final_probs = cls_to_probs(final_probs, race_ids)

    df["win_prob"] = final_probs
    return df


def build_race_objects(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group rows into structured race records."""
    log.info("Grouping entries into structured race objects...")
    races_dict = {}
    
    # Sort chronologically
    df = df.sort_values(by=["date", "race_id", "post_position"])

    for row in df.itertuples():
        rid = row.race_id
        if rid not in races_dict:
            races_dict[rid] = {
                "race_id": rid,
                "date": str(row.date),
                "entries": [],
            }
        
        h_name = getattr(row, "horse_name", f"Horse_{row.post_position}")
        races_dict[rid]["entries"].append({
            "post_position": int(row.post_position),
            "horse_name": h_name,
            "horse_name_jp": getattr(row, "horse_name_jp", None) or h_name,
            "odds": float(row.odds_win),
            "win_prob": float(row.win_prob),
            "finish_pos": int(row.finish_pos) if pd.notna(row.finish_pos) else None,
        })

    return list(races_dict.values())


def evaluate_ticket(
    ticket_type: str,
    selection: List[int],
    stake: int,
    market_odds: float,
    finish_map: Dict[int, int],
    odds_map: Dict[int, float],
) -> Tuple[bool, int]:
    """Evaluate ticket outcome and calculate return."""
    if stake <= 0:
        return False, 0

    if ticket_type in ("win", "単勝"):
        pp = selection[0]
        if finish_map.get(pp) == 1:
            win_odds = odds_map.get(pp, market_odds)
            return True, int(math.floor(stake * win_odds))
        return False, 0

    elif ticket_type in ("place", "複勝"):
        pp = selection[0]
        pos = finish_map.get(pp)
        if pos is not None and 1 <= pos <= 3:
            # Place payout estimated at ~0.30x win odds (JRA standard payout curve)
            place_odds = max(1.1, round(odds_map.get(pp, market_odds) * 0.30, 1))
            return True, int(math.floor(stake * place_odds))
        return False, 0

    elif ticket_type in ("exacta", "馬単"):
        p1, p2 = selection[0], selection[1]
        if finish_map.get(p1) == 1 and finish_map.get(p2) == 2:
            return True, int(math.floor(stake * market_odds))
        return False, 0

    elif ticket_type in ("quinella", "馬連"):
        p1, p2 = selection[0], selection[1]
        pos1, pos2 = finish_map.get(p1), finish_map.get(p2)
        if pos1 is not None and pos2 is not None and {pos1, pos2} == {1, 2}:
            return True, int(math.floor(stake * market_odds))
        return False, 0

    elif ticket_type in ("wide", "ワイド"):
        p1, p2 = selection[0], selection[1]
        pos1, pos2 = finish_map.get(p1), finish_map.get(p2)
        if pos1 is not None and pos2 is not None and pos1 <= 3 and pos2 <= 3:
            return True, int(math.floor(stake * market_odds))
        return False, 0

    elif ticket_type in ("trio", "三連複"):
        p1, p2, p3 = selection[0], selection[1], selection[2]
        pos1, pos2, pos3 = finish_map.get(p1), finish_map.get(p2), finish_map.get(p3)
        if pos1 is not None and pos2 is not None and pos3 is not None and {pos1, pos2, pos3} == {1, 2, 3}:
            return True, int(math.floor(stake * market_odds))
        return False, 0

    return False, 0


def run_strategy_simulation(
    races: List[Dict[str, Any]],
    strategy_id: str,
    budget: int = 1000,
    min_ev: float = 0.05,
    min_prob: float = 0.05,
    max_odds: float = 40.0,
    initial_bankroll: int = 100_000,
    kelly_fraction: float = 0.25,
) -> BenchmarkResult:
    """Simulate a given betting strategy across all chronological races."""
    total_staked = 0
    total_payout = 0
    total_tickets = 0
    winning_tickets = 0
    races_bet = 0
    winning_races = 0

    bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    max_dd_yen = 0
    max_dd_pct = 0.0

    pnl_per_bet = []
    daily_pnl = {}
    odds_bet = []
    ev_bet = []

    mode_counts = {}
    mode_profits = {}
    mode_stakes = {}

    all_dates = sorted(list({r["date"] for r in races}))

    for r in races:
        date_str = r["date"]
        entries = r["entries"]
        if len(entries) < 4:
            continue

        finish_map = {e["post_position"]: e["finish_pos"] for e in entries if e["finish_pos"] is not None}
        odds_map = {e["post_position"]: e["odds"] for e in entries}

        pricing = calculate_pricing_breakdown(entries)

        # -------------------------------------------------------------
        # Strategy Dispatch
        # -------------------------------------------------------------
        tickets_to_bet = []
        chosen_mode = strategy_id

        if strategy_id == "pure_win_flat":
            eligible = [h for h in pricing if h.ev >= min_ev and h.win_prob >= min_prob and h.market_odds <= max_odds]
            if eligible:
                top_v = eligible[0]
                tickets_to_bet.append({
                    "type": "win",
                    "selection": [top_v.post_position],
                    "stake": budget,
                    "odds": top_v.market_odds,
                    "ev": top_v.ev,
                })

        elif strategy_id == "pure_win_kelly":
            eligible = [h for h in pricing if h.ev >= min_ev and h.win_prob >= min_prob and h.market_odds <= max_odds]
            if eligible:
                top_v = eligible[0]
                b = top_v.market_odds - 1.0
                p = top_v.win_prob
                q = 1.0 - p
                f_star = max(0.0, (b * p - q) / b) if b > 0 else 0.0
                stake = int(math.floor((bankroll * f_star * kelly_fraction) / 100.0) * 100)
                stake = min(max(100, stake), int(bankroll * 0.05)) # max 5% bankroll cap
                tickets_to_bet.append({
                    "type": "win",
                    "selection": [top_v.post_position],
                    "stake": stake,
                    "odds": top_v.market_odds,
                    "ev": top_v.ev,
                })

        elif strategy_id == "dual_dutching":
            eligible = [h for h in pricing if h.ev >= min_ev and h.win_prob >= min_prob and h.market_odds <= max_odds]
            if len(eligible) >= 2 and eligible[1].ev >= 0.08:
                v1, v2 = eligible[0], eligible[1]
                s1 = int(math.floor((budget * 0.65) / 100.0) * 100)
                s2 = budget - s1
                tickets_to_bet.append({
                    "type": "win", "selection": [v1.post_position], "stake": s1, "odds": v1.market_odds, "ev": v1.ev
                })
                tickets_to_bet.append({
                    "type": "win", "selection": [v2.post_position], "stake": s2, "odds": v2.market_odds, "ev": v2.ev
                })
            elif eligible:
                top_v = eligible[0]
                tickets_to_bet.append({
                    "type": "win", "selection": [top_v.post_position], "stake": budget, "odds": top_v.market_odds, "ev": top_v.ev
                })

        elif strategy_id == "balanced_hybrid":
            plan = construct_staking_plan(pricing, budget=budget, strategy_mode="hybrid")
            if plan.portfolio_tickets and plan.portfolio_tickets[0].ticket_type_en != "pass":
                for t in plan.portfolio_tickets:
                    tickets_to_bet.append({
                        "type": t.ticket_type_en,
                        "selection": t.selection,
                        "stake": t.stake,
                        "odds": t.market_odds or t.fair_odds,
                        "ev": t.ev or 0.0,
                    })

        elif strategy_id == "place_wide_hedge":
            eligible = [h for h in pricing if h.ev >= min_ev and h.market_odds <= max_odds]
            if eligible:
                top_v = eligible[0]
                fav_list = [h for h in pricing if h.is_favorite]
                fav = fav_list[0] if fav_list else pricing[0]
                # 60% Place + 40% Wide with Fav
                s_place = int(math.floor((budget * 0.60) / 100.0) * 100)
                s_wide = budget - s_place
                tickets_to_bet.append({
                    "type": "place", "selection": [top_v.post_position], "stake": s_place, "odds": top_v.market_odds, "ev": top_v.ev
                })
                if fav.post_position != top_v.post_position:
                    tickets_to_bet.append({
                        "type": "wide", "selection": sorted([top_v.post_position, fav.post_position]), "stake": s_wide, "odds": top_v.market_odds * 0.35, "ev": top_v.ev
                    })

        elif strategy_id == "adaptive_router":
            chosen_mode, reason = select_adaptive_strategy(pricing)
            if chosen_mode != "pass":
                plan = construct_staking_plan(pricing, budget=budget, strategy_mode=chosen_mode)
                if plan.portfolio_tickets and plan.portfolio_tickets[0].ticket_type_en != "pass":
                    for t in plan.portfolio_tickets:
                        tickets_to_bet.append({
                            "type": t.ticket_type_en,
                            "selection": t.selection,
                            "stake": t.stake,
                            "odds": t.market_odds or t.fair_odds,
                            "ev": t.ev or 0.0,
                        })
            mode_counts[chosen_mode] = mode_counts.get(chosen_mode, 0) + 1

        # -------------------------------------------------------------
        # Settlement
        # -------------------------------------------------------------
        race_staked = 0
        race_payout = 0
        race_won = False

        for t in tickets_to_bet:
            won, payout = evaluate_ticket(
                t["type"], t["selection"], t["stake"], t["odds"], finish_map, odds_map
            )
            total_tickets += 1
            if won:
                winning_tickets += 1
                race_won = True
            race_staked += t["stake"]
            race_payout += payout
            odds_bet.append(t["odds"])
            ev_bet.append(t["ev"])

            # Per-bet PnL
            pnl_per_bet.append(payout - t["stake"])

        if race_staked > 0:
            races_bet += 1
            if race_won:
                winning_races += 1
            total_staked += race_staked
            total_payout += race_payout
            race_profit = race_payout - race_staked

            bankroll += race_profit
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
            dd = peak_bankroll - bankroll
            dd_pct = (dd / peak_bankroll) if peak_bankroll > 0 else 0.0
            if dd > max_dd_yen:
                max_dd_yen = dd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

            daily_pnl[date_str] = daily_pnl.get(date_str, 0) + race_profit

            if strategy_id == "adaptive_router":
                mode_profits[chosen_mode] = mode_profits.get(chosen_mode, 0) + race_profit
                mode_stakes[chosen_mode] = mode_stakes.get(chosen_mode, 0) + race_staked

    # -------------------------------------------------------------
    # Aggregate Metrics Calculation
    # -------------------------------------------------------------
    net_profit = total_payout - total_staked
    roi_pct = (net_profit / total_staked * 100.0) if total_staked > 0 else 0.0
    ticket_hit_rate = (winning_tickets / total_tickets * 100.0) if total_tickets > 0 else 0.0
    race_win_rate = (winning_races / races_bet * 100.0) if races_bet > 0 else 0.0

    # Per-bet Sharpe
    if len(pnl_per_bet) > 10:
        arr_bet = np.array(pnl_per_bet)
        per_bet_sharpe = float(arr_bet.mean() / arr_bet.std()) if arr_bet.std() > 0 else 0.0
    else:
        per_bet_sharpe = 0.0

    # Daily Sharpe (Active Days Only)
    active_daily_returns = np.array(list(daily_pnl.values())) / initial_bankroll
    if len(active_daily_returns) > 5 and active_daily_returns.std() > 0:
        daily_sharpe_active = float((active_daily_returns.mean() / active_daily_returns.std()) * np.sqrt(104))
    else:
        daily_sharpe_active = 0.0

    # Daily Sharpe (Padded with zero for non-betting race days)
    padded_daily_pnl = {d: daily_pnl.get(d, 0) for d in all_dates}
    padded_daily_returns = np.array(list(padded_daily_pnl.values())) / initial_bankroll
    if len(padded_daily_returns) > 5 and padded_daily_returns.std() > 0:
        daily_sharpe_padded = float((padded_daily_returns.mean() / padded_daily_returns.std()) * np.sqrt(104))
    else:
        daily_sharpe_padded = 0.0

    gross_wins = sum(p for p in pnl_per_bet if p > 0)
    gross_losses = abs(sum(p for p in pnl_per_bet if p < 0))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else 999.0

    return BenchmarkResult(
        strategy_name=strategy_id,
        total_races=len(races),
        races_bet=races_bet,
        total_tickets=total_tickets,
        winning_tickets=winning_tickets,
        total_staked=total_staked,
        total_payout=total_payout,
        net_profit=net_profit,
        roi_pct=round(roi_pct, 2),
        ticket_hit_rate=round(ticket_hit_rate, 2),
        race_win_rate=round(race_win_rate, 2),
        max_drawdown_yen=int(max_dd_yen),
        max_drawdown_pct=round(max_dd_pct * 100, 2),
        daily_sharpe_padded=round(daily_sharpe_padded, 2),
        daily_sharpe_active=round(daily_sharpe_active, 2),
        per_bet_sharpe=round(per_bet_sharpe, 3),
        profit_factor=round(profit_factor, 2),
        avg_odds=round(float(np.mean(odds_bet)), 2) if odds_bet else 0.0,
        avg_ev=round(float(np.mean(ev_bet)), 3) if ev_bet else 0.0,
        mode_counts=mode_counts,
        mode_profits=mode_profits,
        mode_stakes=mode_stakes,
    )


def run_full_benchmark(
    test_start_date: Optional[str] = "2024-01-01",
    model_version: str = "retrain_20260822_2143",
    budget: int = 1000,
):
    """Run full benchmark across all strategies and print comparative leaderboard."""
    pred_df = generate_dataset_predictions(
        test_start_date=test_start_date,
        model_version=model_version,
    )
    races = build_race_objects(pred_df)
    log.info(f"Successfully constructed {len(races):,} historical race objects.")

    strategies_to_test = [
        ("pure_win_flat", "⚡ Pure Value Win (Flat ¥1k)"),
        ("pure_win_kelly", "⚡ Pure Value Win (Kelly 0.25)"),
        ("dual_dutching", "🛡️ Dual Dutching (Low Drawdown)"),
        ("balanced_hybrid", "🎯 Balanced Portfolio (Win+Exotics)"),
        ("place_wide_hedge", "🪢 Place / Wide Longshot Hedge"),
        ("adaptive_router", "🧠 Adaptive Strategy Router (AI Optimal)"),
    ]

    results = []
    log.info("\n" + "="*80)
    log.info("RUNNING BENCHMARK SIMULATIONS ACROSS ALL STRATEGIES")
    log.info("="*80)

    for strat_id, label in strategies_to_test:
        log.info(f"Running simulation: {label}...")
        res = run_strategy_simulation(
            races,
            strategy_id=strat_id,
            budget=budget,
            min_ev=0.08,
            min_prob=0.06,
        )
        res.strategy_name = label
        results.append(res)

    # -------------------------------------------------------------
    # Format and Output Scorecard
    # -------------------------------------------------------------
    rows = []
    for r in results:
        rows.append({
            "Strategy": r.strategy_name,
            "Races Bet": f"{r.races_bet:,} / {r.total_races:,} ({r.races_bet/r.total_races*100:.1f}%)",
            "Win Rate": f"{r.race_win_rate:.1f}%",
            "Total Staked": f"¥{r.total_staked:,}",
            "Net Profit": f"{'+' if r.net_profit > 0 else ''}¥{r.net_profit:,}",
            "ROI (%)": f"{'+' if r.roi_pct > 0 else ''}{r.roi_pct:.1f}%",
            "Max DD (%)": f"{r.max_drawdown_pct:.1f}%",
            "Daily Sharpe (Padded)": r.daily_sharpe_padded,
            "Daily Sharpe (Active)": r.daily_sharpe_active,
            "Profit Factor": r.profit_factor,
            "Avg Odds": f"{r.avg_odds:.1f}x",
        })

    df_scorecard = pd.DataFrame(rows)

    print("\n" + "="*120)
    print(f"UMAEDGE HISTORICAL BETTING STRATEGY BENCHMARK ({test_start_date or '2014'} to 2026 — {len(races):,} Races)")
    print("="*120)
    print(df_scorecard.to_string(index=False))
    print("="*120)

    # Adaptive Strategy Breakdown
    adaptive_res = [r for r in results if "Adaptive" in r.strategy_name][0]
    print("\n" + "-"*80)
    print("🧠 ADAPTIVE STRATEGY ROUTER — DECISION BREAKDOWN & SUB-PERFORMANCE")
    print("-"*80)
    total_decisions = sum(adaptive_res.mode_counts.values()) or 1
    for mode, count in sorted(adaptive_res.mode_counts.items(), key=lambda x: x[1], reverse=True):
        staked = adaptive_res.mode_stakes.get(mode, 0)
        profit = adaptive_res.mode_profits.get(mode, 0)
        sub_roi = (profit / staked * 100.0) if staked > 0 else 0.0
        pct = (count / total_decisions) * 100.0
        print(f"  • {mode:<12}: {count:>5} races ({pct:>5.1f}%) | Staked: ¥{staked:>10,} | Profit: {'+' if profit>0 else ''}¥{profit:>9,} | ROI: {sub_roi:>+6.1f}%")
    print("-"*80)

    # Save to CSV
    os.makedirs("results", exist_ok=True)
    out_path = "results/benchmark_all_strategies.csv"
    df_scorecard.to_csv(out_path, index=False)
    log.info(f"\nSaved full benchmark leaderboard to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UmaEdge Strategy Benchmark")
    parser.add_argument("--test-start", type=str, default="2024-01-01", help="Start date for test window (default: 2024-01-01)")
    parser.add_argument("--full-dataset", action="store_true", help="Run benchmark across entire database (2014 to 2026)")
    parser.add_argument("--budget", type=int, default=1000, help="Standard budget per race (default: 1000)")
    parser.add_argument("--model-version", type=str, default="retrain_20260822_2143", help="Model version to benchmark")
    args = parser.parse_args()

    test_start = None if args.full_dataset else args.test_start
    run_full_benchmark(
        test_start_date=test_start,
        model_version=args.model_version,
        budget=args.budget,
    )
