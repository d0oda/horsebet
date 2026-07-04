"""
UmaEdge — Exotic (Trio / 三連複) Backtesting (Sprint 3.4).

Backtests trio tickets using model probabilities and actual race results.
For each race: build top-N trio combinations from model probs, check if
actual top-3 result matches, and compute ROI.

Usage:
    python -m models.backtest_trio
    python -m models.backtest_trio --budget 5000 --top-n 20
"""

import argparse
import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backtest_trio")


JRA_TRIO_TAKE_RATE = 0.275  # JRA take rate for trio (~27.5%)


# ---------------------------------------------------------------------------
# Trio Backtest Data
# ---------------------------------------------------------------------------

@dataclass
class TrioTicket:
    """A trio ticket and its outcome."""
    race_id: int
    date: str
    combination: tuple[int, int, int]  # horse_ids sorted
    model_prob: float
    estimated_odds: float
    ev: float
    actual_top3: tuple  # actual top-3 horse IDs
    is_hit: bool
    payout: int
    cost: int = 100


@dataclass
class TrioBacktestResult:
    """Results of a trio backtest."""
    total_races: int = 0
    total_tickets: int = 0
    hits: int = 0
    total_cost: int = 0
    total_payout: int = 0
    roi_pct: float = 0.0
    hit_rate: float = 0.0
    avg_ev: float = 0.0
    avg_odds: float = 0.0
    tickets: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core Trio Backtester
# ---------------------------------------------------------------------------

def compute_trio_prob(
    horse_probs: dict[int, float],
    combo: tuple[int, int, int],
) -> float:
    """
    Estimate trio probability from win probabilities.
    P(trio) ≈ sum of all permutations of these 3 horses finishing top-3.
    Simplified: P(A top-3) × P(B top-3 | A) × P(C top-3 | A,B)
    Approximation: product of their place probabilities.
    """
    p = 1.0
    remaining_prob = 1.0
    for hid in combo:
        wp = horse_probs.get(hid, 0.01)
        # P(horse in top-3) ≈ 3 × win_prob (capped at 0.95)
        place_p = min(wp * 3, 0.95)
        p *= (place_p / remaining_prob) if remaining_prob > 0 else 0
        remaining_prob -= wp  # approximate exclusion
    # Adjust for number of permutations (3! = 6 orderings)
    return min(p * 6, 0.95)


def run_trio_backtest(
    predictions_df: pd.DataFrame,
    budget_per_race: int = 3000,
    top_n_tickets: int = 10,
    min_ev: float = 0.0,
) -> TrioBacktestResult:
    """
    Backtest trio tickets on historical races.

    Args:
        predictions_df: DataFrame with race_id, entry_id, horse_id,
                         win_prob, odds_win, finish_pos, date
        budget_per_race: Max spend per race in yen
        top_n_tickets: Max tickets to buy per race
        min_ev: Minimum EV to include ticket
    """
    result = TrioBacktestResult()

    races = predictions_df.groupby("race_id")
    result.total_races = len(races)

    for race_id, race_df in races:
        if len(race_df) < 3:
            continue

        # Build win probability map
        horse_probs = {}
        horse_names = {}
        for _, row in race_df.iterrows():
            hid = row.get("horse_id") or row.get("entry_id")
            horse_probs[hid] = row.get("win_prob", 0.01)
            horse_names[hid] = row.get("horse_name", "?")

        # Actual top-3
        finished = race_df[race_df["finish_pos"].notna()].sort_values("finish_pos")
        if len(finished) < 3:
            continue

        actual_top3 = tuple(sorted([
            finished.iloc[0].get("horse_id") or finished.iloc[0].get("entry_id"),
            finished.iloc[1].get("horse_id") or finished.iloc[1].get("entry_id"),
            finished.iloc[2].get("horse_id") or finished.iloc[2].get("entry_id"),
        ]))

        # Generate all combos of top-6 horses (by model prob)
        sorted_horses = sorted(horse_probs.items(), key=lambda x: -x[1])
        top_horses = [hid for hid, _ in sorted_horses[:min(8, len(sorted_horses))]]
        combos = list(combinations(top_horses, 3))

        # Score each combo
        scored = []
        for combo in combos:
            combo_sorted = tuple(sorted(combo))
            prob = compute_trio_prob(horse_probs, combo_sorted)
            estimated_odds = (1.0 / prob) * (1 - JRA_TRIO_TAKE_RATE) if prob > 0 else 0
            ev = prob * estimated_odds - 1.0

            if ev >= min_ev:
                scored.append((combo_sorted, prob, estimated_odds, ev))

        # Buy top N tickets by EV
        scored.sort(key=lambda x: -x[3])
        tickets_to_buy = scored[:top_n_tickets]

        max_tickets = budget_per_race // 100
        tickets_to_buy = tickets_to_buy[:max_tickets]

        date_str = str(race_df.iloc[0].get("date", ""))

        for combo_sorted, prob, est_odds, ev in tickets_to_buy:
            is_hit = combo_sorted == actual_top3
            payout = int(100 * est_odds) if is_hit else 0

            ticket = TrioTicket(
                race_id=race_id,
                date=date_str,
                combination=combo_sorted,
                model_prob=prob,
                estimated_odds=est_odds,
                ev=ev,
                actual_top3=actual_top3,
                is_hit=is_hit,
                payout=payout,
            )
            result.tickets.append(ticket)

    # Compute summary
    result.total_tickets = len(result.tickets)
    result.hits = sum(1 for t in result.tickets if t.is_hit)
    result.total_cost = result.total_tickets * 100
    result.total_payout = sum(t.payout for t in result.tickets)
    result.roi_pct = ((result.total_payout - result.total_cost) / result.total_cost * 100) if result.total_cost > 0 else 0
    result.hit_rate = (result.hits / result.total_tickets * 100) if result.total_tickets > 0 else 0
    result.avg_ev = np.mean([t.ev for t in result.tickets]) if result.tickets else 0
    result.avg_odds = np.mean([t.estimated_odds for t in result.tickets]) if result.tickets else 0

    return result


def print_trio_report(result: TrioBacktestResult):
    """Print formatted trio backtest report."""
    print(f"\n{'=' * 60}")
    print("  UmaEdge — Trio (三連複) Backtest Report")
    print(f"{'=' * 60}")
    print(f"  Races analysed:     {result.total_races:>8}")
    print(f"  Tickets bought:     {result.total_tickets:>8}")
    print(f"  Hits:               {result.hits:>8}")
    print(f"  Hit rate:           {result.hit_rate:>7.1f}%")
    print(f"  Avg EV per ticket:  {result.avg_ev:>7.3f}")
    print(f"  Avg est. odds:      {result.avg_odds:>7.1f}x")
    print("-" * 60)
    print(f"  Total cost:         ¥{result.total_cost:>10,}")
    print(f"  Total payout:       ¥{result.total_payout:>10,}")
    profit = result.total_payout - result.total_cost
    print(f"  Profit:             ¥{profit:>10,}")
    print(f"  ROI:                {result.roi_pct:>7.1f}%")
    print(f"{'=' * 60}")

    if result.hits > 0:
        print(f"\n  Winning tickets:")
        for t in result.tickets:
            if t.is_hit:
                print(f"    Race {t.race_id} ({t.date}): "
                      f"{t.combination} odds={t.estimated_odds:.1f}x "
                      f"payout=¥{t.payout:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Trio Backtest")
    parser.add_argument("--budget", type=int, default=3000, help="Budget per race in yen")
    parser.add_argument("--top-n", type=int, default=10, help="Max tickets per race")
    parser.add_argument("--min-ev", type=float, default=0.0, help="Minimum EV to include ticket")
    args = parser.parse_args()

    # Build predictions from feature pipeline
    from models.features import FeatureBuilder
    from models.train import prepare_data, train_lightgbm, train_xgboost, ensemble_predict
    import xgboost as xgb

    log.info("Building features...")
    fb = FeatureBuilder()
    df = fb.build_features_all()
    if df.empty:
        print("No data — run scraper first")
        return

    log.info("Preparing data...")
    X_train, y_train, X_val, y_val, feature_cols, _, _ = prepare_data(
        df, target="target_win", val_date="2025-01-01"
    )

    log.info("Training models...")
    lgb_model, lgb_preds = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols)
    xgb_model, xgb_preds = train_xgboost(X_train, y_train, X_val, y_val, feature_cols)

    # Predict on validation set
    lgb_val_preds = lgb_model.predict(X_val)
    xgb_val_preds = xgb_model.predict(xgb.DMatrix(X_val, feature_names=feature_cols))
    preds = ensemble_predict(lgb_val_preds, xgb_val_preds)

    # Build validation dataframe with predictions
    val_mask = df["date"] >= "2025-01-01"
    val_df = df[val_mask].copy()
    val_df["win_prob"] = preds

    result = run_trio_backtest(
        val_df,
        budget_per_race=args.budget,
        top_n_tickets=args.top_n,
        min_ev=args.min_ev,
    )
    print_trio_report(result)


if __name__ == "__main__":
    main()
