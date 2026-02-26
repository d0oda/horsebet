"""
UmaEdge — Losing Bet Failure Mode Analysis (Sprint 3.3).

Categorises losing bets from a backtest into failure modes:
  (a) Model overconfident — model_prob > 2× market_prob
  (b) No edge / fair odds — model_prob within ±50% of market_prob
  (c) Variance / bad luck — horse finished 2nd or 3rd (close miss)

Usage:
    from models.analyse_bets import analyse_losing_bets
    analyse_losing_bets(backtest_result)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("analyse_bets")


# ---------------------------------------------------------------------------
# Failure Mode Classification
# ---------------------------------------------------------------------------

@dataclass
class FailureMode:
    """Summary of a failure category."""
    name: str
    count: int
    total_loss: int
    avg_loss: float
    avg_model_prob: float
    avg_market_prob: float
    avg_odds: float
    examples: list  # top-3 worst losses


def classify_bet(model_prob: float, market_prob: float, finish_pos: Optional[int]) -> str:
    """
    Classify a losing bet into a failure mode.

    Returns:
        One of: 'overconfident', 'no_edge', 'variance', 'unknown'
    """
    if finish_pos is not None and finish_pos in (2, 3):
        return "variance"  # close call — horse was competitive

    ratio = model_prob / market_prob if market_prob > 0 else float("inf")

    if ratio > 2.0:
        return "overconfident"  # model vastly overestimated
    elif ratio < 1.5:
        return "no_edge"  # no real edge, odds were fair
    else:
        return "unknown"  # marginal edge, just didn't hit


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_losing_bets(backtest_result) -> dict[str, FailureMode]:
    """
    Analyse losing bets from a BacktestResult.

    Args:
        backtest_result: A BacktestResult with .bets list

    Returns:
        Dict of failure_mode_name -> FailureMode summary
    """
    losing = [b for b in backtest_result.bets if b.profit < 0]

    if not losing:
        print("\n  No losing bets to analyse!")
        return {}

    # Classify each loser
    classified = {}
    for b in losing:
        mode = classify_bet(b.model_prob, b.market_prob, b.finish_pos)
        classified.setdefault(mode, []).append(b)

    # Build summaries
    modes = {}
    for mode_name, bets in classified.items():
        losses = [b.profit for b in bets]
        modes[mode_name] = FailureMode(
            name=mode_name,
            count=len(bets),
            total_loss=sum(losses),
            avg_loss=np.mean(losses),
            avg_model_prob=np.mean([b.model_prob for b in bets]),
            avg_market_prob=np.mean([b.market_prob for b in bets]),
            avg_odds=np.mean([b.odds for b in bets]),
            examples=sorted(bets, key=lambda b: b.profit)[:3],
        )

    # Print report
    _print_analysis(modes, total_losing=len(losing))

    return modes


def _print_analysis(modes: dict[str, FailureMode], total_losing: int):
    """Print a formatted failure mode analysis report."""
    labels = {
        "overconfident": "🔴 Model Overconfident (P_model > 2× P_market)",
        "no_edge": "🟡 No Edge / Fair Odds (P_model ≈ P_market)",
        "variance": "🟢 Variance / Bad Luck (finished 2nd/3rd)",
        "unknown": "⚪ Marginal Edge (1.5–2× ratio)",
    }

    print(f"\n{'=' * 60}")
    print("  UmaEdge — Losing Bet Analysis")
    print(f"{'=' * 60}")
    print(f"  Total losing bets:  {total_losing}")
    print()

    for mode_name in ["overconfident", "no_edge", "variance", "unknown"]:
        if mode_name not in modes:
            continue
        m = modes[mode_name]
        label = labels.get(mode_name, mode_name)
        pct = m.count / total_losing * 100

        print(f"  {label}")
        print(f"    Count:        {m.count:>4} ({pct:.0f}%)")
        print(f"    Total loss:   ¥{m.total_loss:>8,}")
        print(f"    Avg loss:     ¥{m.avg_loss:>8,.0f}")
        print(f"    Avg P(model): {m.avg_model_prob:>6.1%}")
        print(f"    Avg P(mkt):   {m.avg_market_prob:>6.1%}")
        print(f"    Avg odds:     {m.avg_odds:>6.1f}x")

        if m.examples:
            print(f"    Worst losses:")
            for ex in m.examples:
                print(
                    f"      {ex.horse_name:<14} P={ex.model_prob:.1%} "
                    f"vs mkt={ex.market_prob:.1%}  odds={ex.odds:.1f}x  "
                    f"finish={ex.finish_pos or '?'}  ¥{ex.profit:,}"
                )
        print()

    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="UmaEdge — Losing Bet Failure Mode Analysis"
    )
    parser.add_argument(
        "--ev-threshold", type=float, default=0.05,
        help="EV threshold for bet selection (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--calibration", type=str, default="isotonic",
        choices=["none", "platt", "isotonic"],
        help="Calibration method (default: isotonic)",
    )
    args = parser.parse_args()

    from models.test_2025 import run_2025_evaluation

    log.info(f"Running evaluation with EV threshold {args.ev_threshold:.0%}...")
    result = run_2025_evaluation(
        ev_threshold=args.ev_threshold,
        calibration_method=args.calibration,
    )

    if result and result.bets:
        modes = analyse_losing_bets(result)
        if not modes:
            log.info("No losing bets — all bets were winners!")
    else:
        log.warning("No bets found at the given EV threshold.")


if __name__ == "__main__":
    main()
