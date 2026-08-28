"""
UmaEdge — Deep Hybrid Betting Portfolio Optimizer.

Tests and optimizes:
1. Pure Value Win vs Dutching (2 runners)
2. Hybrid Multi-Ticket Staking (Win + Elite Positive-EV Exotics)
3. Minimum EV thresholds for Core vs Exotic legs
4. Capital allocation weights (80/20 vs 70/30 vs 60/40 vs 100/0)
5. Risk-adjusted metrics (Sharpe, Calmar, Max Drawdown %, Monthly Win Rate)
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from models.evaluate_betting_strategies import (
    load_historical_eval_data,
    evaluate_ticket_payout,
    StrategyMetrics,
    TAKEOUT_MAP,
)
from models.betting_engine import (
    calculate_pricing_breakdown,
    JointFinishModel,
)


def run_hybrid_portfolio_experiment(
    races: List[Dict[str, Any]],
    name: str,
    min_core_ev: float = 0.10,
    min_exotic_ev: float = 0.15,
    min_win_prob: float = 0.05,
    max_odds: float = 35.0,
    win_weight: float = 0.75,
    dutch_second_runner: bool = False,
    budget: int = 1000,
) -> StrategyMetrics:
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

        pricing = calculate_pricing_breakdown(entries)
        probs_by_pp = {h.post_position: h.win_prob for h in pricing}
        model = JointFinishModel(probs_by_pp)

        mkt_probs_by_pp = {h.post_position: 1.0 / max(1.01, h.market_odds) for h in pricing}
        mkt_model = JointFinishModel(mkt_probs_by_pp, gamma=model.gamma)

        # 1. Candidate value runners
        value_runners = [
            h for h in pricing
            if h.ev >= min_core_ev
            and h.win_prob >= min_win_prob
            and h.market_odds <= max_odds
        ]

        if not value_runners:
            continue

        anchor = value_runners[0]
        fav_horses = [h for h in pricing if h.is_favorite]
        favorite = fav_horses[0] if fav_horses else pricing[0]

        contenders = [h for h in pricing if h.post_position != anchor.post_position]
        c1 = contenders[0] if len(contenders) > 0 else anchor
        c2 = contenders[1] if len(contenders) > 1 else c1

        # 2. Build candidate tickets
        race_tickets = []

        # Core Win Bet on Anchor
        race_tickets.append({
            "type_en": "win",
            "selection": [anchor.post_position],
            "display": f"No. {anchor.post_position} 単勝",
            "odds": anchor.market_odds,
            "ev": anchor.ev,
            "is_core": True,
            "weight": win_weight if not dutch_second_runner else (win_weight * 0.65),
        })

        # Dutching: Second value runner Win bet
        if dutch_second_runner and len(value_runners) > 1:
            second_v = value_runners[1]
            race_tickets.append({
                "type_en": "win",
                "selection": [second_v.post_position],
                "display": f"No. {second_v.post_position} 単勝 (Dutch)",
                "odds": second_v.market_odds,
                "ev": second_v.ev,
                "is_core": True,
                "weight": win_weight * 0.35,
            })

        # Exotic Candidates with strict min_exotic_ev
        exotic_candidates = []

        # Exacta: Anchor -> Contender / Fav
        if c1.post_position != anchor.post_position:
            ex_p = model.exacta_prob(anchor.post_position, c1.post_position)
            ex_mkt_p = mkt_model.exacta_prob(anchor.post_position, c1.post_position)
            ex_odds = round((1.0 - TAKEOUT_MAP["exacta"]) / max(0.0001, ex_mkt_p), 1)
            ex_ev = (ex_p * ex_odds) - 1.0
            if ex_ev >= min_exotic_ev:
                exotic_candidates.append({
                    "type_en": "exacta",
                    "selection": [anchor.post_position, c1.post_position],
                    "display": f"{anchor.post_position} → {c1.post_position} 馬単",
                    "odds": ex_odds,
                    "ev": ex_ev,
                    "prob": ex_p,
                    "is_core": False,
                })

        # Quinella: Anchor - Target Partner
        target_partner = favorite if favorite.post_position != anchor.post_position else c1
        if target_partner.post_position != anchor.post_position:
            q_p = model.quinella_prob(anchor.post_position, target_partner.post_position)
            q_mkt_p = mkt_model.quinella_prob(anchor.post_position, target_partner.post_position)
            q_odds = round((1.0 - TAKEOUT_MAP["quinella"]) / max(0.0001, q_mkt_p), 1)
            q_ev = (q_p * q_odds) - 1.0
            if q_ev >= min_exotic_ev:
                exotic_candidates.append({
                    "type_en": "quinella",
                    "selection": sorted([anchor.post_position, target_partner.post_position]),
                    "display": f"{min(anchor.post_position, target_partner.post_position)}–{max(anchor.post_position, target_partner.post_position)} 馬連",
                    "odds": q_odds,
                    "ev": q_ev,
                    "prob": q_p,
                    "is_core": False,
                })

        # Wide: Anchor - Fav
        if favorite.post_position != anchor.post_position:
            w_p = model.wide_prob(anchor.post_position, favorite.post_position)
            w_mkt_p = mkt_model.wide_prob(anchor.post_position, favorite.post_position)
            w_odds = round((1.0 - TAKEOUT_MAP["wide"]) / max(0.0001, w_mkt_p), 1)
            w_ev = (w_p * w_odds) - 1.0
            if w_ev >= min_exotic_ev:
                exotic_candidates.append({
                    "type_en": "wide",
                    "selection": sorted([anchor.post_position, favorite.post_position]),
                    "display": f"{min(anchor.post_position, favorite.post_position)}–{max(anchor.post_position, favorite.post_position)} ワイド",
                    "odds": w_odds,
                    "ev": w_ev,
                    "prob": w_p,
                    "is_core": False,
                })

        # Trio: Anchor - Fav - Secondary
        other_c = c1 if c1.post_position != favorite.post_position else c2
        trio_pps = sorted(list({anchor.post_position, favorite.post_position, other_c.post_position}))
        if len(trio_pps) == 3:
            tr_p = model.trio_prob(trio_pps[0], trio_pps[1], trio_pps[2])
            tr_mkt_p = mkt_model.trio_prob(trio_pps[0], trio_pps[1], trio_pps[2])
            tr_odds = round((1.0 - TAKEOUT_MAP["trio"]) / max(0.0001, tr_mkt_p), 1)
            tr_ev = (tr_p * tr_odds) - 1.0
            if tr_ev >= min_exotic_ev:
                exotic_candidates.append({
                    "type_en": "trio",
                    "selection": trio_pps,
                    "display": f"{trio_pps[0]}–{trio_pps[1]}–{trio_pps[2]} 三連複",
                    "odds": tr_odds,
                    "ev": tr_ev,
                    "prob": tr_p,
                    "is_core": False,
                })

        # Add top qualified exotics
        exotic_candidates.sort(key=lambda x: x["ev"], reverse=True)
        qualified_exotics = exotic_candidates[:2]

        if not qualified_exotics or win_weight >= 1.0:
            # 100% on Core Win
            for t in race_tickets:
                t["weight"] = 1.0 / len(race_tickets)
        else:
            exotic_weight_pool = 1.0 - win_weight
            ex_score_total = sum(max(0.01, ex["prob"] * (1.0 + ex["ev"])) for ex in qualified_exotics)
            for ex in qualified_exotics:
                score = max(0.01, ex["prob"] * (1.0 + ex["ev"]))
                ex["weight"] = exotic_weight_pool * (score / ex_score_total)
                race_tickets.append(ex)

        # 3. Discrete ¥100 Allocation
        total_weight = sum(t["weight"] for t in race_tickets)
        stakes = []
        for t in race_tickets:
            norm_w = t["weight"] / total_weight
            stk = max(100, int(math.floor((budget * norm_w) / 100.0) * 100))
            stakes.append(stk)

        # Rebalance to budget
        while sum(stakes) > budget:
            reducible = [i for i, s in enumerate(stakes) if s > 100]
            if reducible:
                idx = min(reducible, key=lambda i: (race_tickets[i]["weight"], -i))
                stakes[idx] -= 100
            else:
                stakes.pop()
                race_tickets.pop()

        diff = budget - sum(stakes)
        if diff > 0 and stakes:
            stakes[0] += diff  # Add remainder to primary win ticket

        # Evaluate outcome
        race_staked = 0
        race_payout = 0
        race_won = False

        for i, t in enumerate(race_tickets):
            stk = stakes[i]
            won, payout = evaluate_ticket_payout(
                t["type_en"], t["selection"], stk, t["odds"], finish_map, odds_map
            )
            total_tickets += 1
            if won:
                winning_tickets += 1
                race_won = True
            race_staked += stk
            race_payout += payout
            odds_bet_list.append(t["odds"])
            ev_bet_list.append(t["ev"])

        if race_staked > 0:
            races_bet += 1
            if race_won:
                winning_races += 1
            total_staked += race_staked
            total_payout += race_payout
            profit = race_payout - race_staked
            pnl_history.append(profit)

            running_bankroll += profit
            if running_bankroll > peak_bankroll:
                peak_bankroll = running_bankroll
            dd = peak_bankroll - running_bankroll
            dd_pct = dd / peak_bankroll if peak_bankroll > 0 else 0.0
            if dd > max_drawdown_yen:
                max_drawdown_yen = dd
            if dd_pct > max_drawdown_pct:
                max_drawdown_pct = dd_pct

            date_str = r["date"]
            daily_pnl[date_str] = daily_pnl.get(date_str, 0) + profit

    net_profit = total_payout - total_staked
    roi_pct = (net_profit / total_staked * 100.0) if total_staked > 0 else 0.0
    hit_rate = (winning_tickets / total_tickets * 100.0) if total_tickets > 0 else 0.0
    race_hit_rate = (winning_races / races_bet * 100.0) if races_bet > 0 else 0.0

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


def run_hybrid_optimization_sweep():
    races = load_historical_eval_data()
    print(f"Loaded {len(races)} historical races for hybrid optimization sweep.\n")

    experiments = []

    # 1. Pure Win Baseline at different EV thresholds
    for ev in [0.05, 0.10, 0.15, 0.20]:
        experiments.append(run_hybrid_portfolio_experiment(
            races,
            name=f"100% Pure Win (EV >= {int(ev*100)}%)",
            min_core_ev=ev,
            win_weight=1.0,
            dutch_second_runner=False,
        ))

    # 2. Dutching (Dual Value Runners)
    for ev in [0.10, 0.15, 0.20]:
        experiments.append(run_hybrid_portfolio_experiment(
            races,
            name=f"Dutching 2 Runners (EV >= {int(ev*100)}%)",
            min_core_ev=ev,
            win_weight=1.0,
            dutch_second_runner=True,
        ))

    # 3. Hybrid Portfolio: 80% Win / 20% Elite Exotics (EV >= 15%)
    for core_ev in [0.10, 0.15, 0.20]:
        experiments.append(run_hybrid_portfolio_experiment(
            races,
            name=f"Hybrid 80/20 (Core EV>={int(core_ev*100)}%, Exot EV>=15%)",
            min_core_ev=core_ev,
            min_exotic_ev=0.15,
            win_weight=0.80,
            dutch_second_runner=False,
        ))

    # 4. Hybrid Portfolio: 70% Win / 30% Elite Exotics (EV >= 20%)
    for core_ev in [0.10, 0.15, 0.20]:
        experiments.append(run_hybrid_portfolio_experiment(
            races,
            name=f"Hybrid 70/30 (Core EV>={int(core_ev*100)}%, Exot EV>=20%)",
            min_core_ev=core_ev,
            min_exotic_ev=0.20,
            win_weight=0.70,
            dutch_second_runner=False,
        ))

    # 5. Hybrid Dutching + Elite Exotics
    experiments.append(run_hybrid_portfolio_experiment(
        races,
        name="Hybrid Dutching 75/25 (Core EV>=15%, Exot EV>=15%)",
        min_core_ev=0.15,
        min_exotic_ev=0.15,
        win_weight=0.75,
        dutch_second_runner=True,
    ))

    df = pd.DataFrame([
        {
            "Strategy": r.name,
            "Races Bet": f"{r.races_bet}/{r.total_races} ({r.races_bet/r.total_races*100:.1f}%)",
            "Race Hit Rate": f"{r.race_hit_rate}%",
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
        for r in experiments
    ])

    print("="*120)
    print("HYBRID BETTING PORTFOLIO OPTIMIZATION SWEEP")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120)

    df.to_csv("data/hybrid_portfolio_optimization_results.csv", index=False)
    print("\nSaved hybrid optimization results to data/hybrid_portfolio_optimization_results.csv")


if __name__ == "__main__":
    run_hybrid_optimization_sweep()
