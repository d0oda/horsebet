"""
UmaEdge — Post-Scrape Evaluation Runner.

Runs all evaluation steps from post_scrape_plan.md in sequence:
  1. Verify data volume (race counts per year)
  2. Full EV sweep with isotonic calibration
  3. Odds-free EV sweep (Hypothesis H2)
  4. Losing bet analysis at 5% threshold
  5. Trio exotic backtest

Prints a final decision matrix comparing results against pass criteria.

Usage:
    python -m models.run_evaluation
    python -m models.run_evaluation --skip-trio
    python -m models.run_evaluation --only 1      # only verify data
    python -m models.run_evaluation --only 3      # only odds-free eval
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluation")


def banner(title: str):
    print(f"\n{'#' * 70}")
    print(f"#  {title}")
    print(f"{'#' * 70}\n")


# ---------------------------------------------------------------------------
# Step 1 — Verify Data Volume
# ---------------------------------------------------------------------------

def step_verify_data() -> dict:
    """Check race counts per year. Returns dict of year -> count."""
    banner("Step 1 — Verify Data Volume")

    from scraper.db import get_session
    from sqlalchemy import text

    with get_session() as session:
        rows = session.execute(text("""
            SELECT
                EXTRACT(YEAR FROM date)::int AS year,
                COUNT(*) AS races
            FROM races
            WHERE field_size IS NOT NULL
            GROUP BY EXTRACT(YEAR FROM date)
            ORDER BY year
        """)).fetchall()

    counts = {}
    total = 0
    print(f"  {'Year':<6} {'Races':>6}  {'Target':>7}  {'Status':>6}")
    print("  " + "-" * 32)

    targets = {2022: 500, 2023: 500, 2024: 500, 2025: 100}

    for year, races in rows:
        counts[int(year)] = races
        total += races
        target = targets.get(int(year), 0)
        status = "✅" if races >= target else "⚠️"
        print(f"  {int(year):<6} {races:>6}  {target:>7}  {status:>6}")

    print("  " + "-" * 32)
    print(f"  {'Total':<6} {total:>6}  {'1600':>7}  {'✅' if total >= 1600 else '⚠️':>6}")

    return counts


# ---------------------------------------------------------------------------
# Step 2 — Full EV Sweep
# ---------------------------------------------------------------------------

def step_ev_sweep() -> Optional[list]:
    """Run EV sweep with isotonic calibration. Returns list of (threshold, result)."""
    banner("Step 2 — Full EV Sweep (with odds, isotonic calibration)")
    from models.test_2025 import run_ev_sweep
    return run_ev_sweep(exclude_odds=False, calibration_method="isotonic")


# ---------------------------------------------------------------------------
# Step 3 — Odds-Free Evaluation
# ---------------------------------------------------------------------------

def step_odds_free() -> Optional[list]:
    """Run odds-free EV sweep. Returns list of (threshold, result)."""
    banner("Step 3 — Odds-Free Evaluation (Hypothesis H2)")
    from models.test_2025 import run_ev_sweep
    return run_ev_sweep(exclude_odds=True, calibration_method="isotonic")


# ---------------------------------------------------------------------------
# Step 4 — Losing Bet Analysis
# ---------------------------------------------------------------------------

def step_losing_bets(ev_threshold: float = 0.05) -> Optional[dict]:
    """Analyse losing bets at the given EV threshold."""
    banner(f"Step 4 — Losing Bet Analysis (EV threshold {ev_threshold:.0%})")

    from models.test_2025 import run_2025_evaluation
    from models.analyse_bets import analyse_losing_bets

    result = run_2025_evaluation(
        ev_threshold=ev_threshold,
        calibration_method="isotonic",
    )

    if result and result.bets:
        modes = analyse_losing_bets(result)
        return modes
    else:
        log.warning("No bets to analyse at this threshold")
        return None


# ---------------------------------------------------------------------------
# Step 5 — Trio Exotic Backtest
# ---------------------------------------------------------------------------

def step_trio_backtest(budget: int = 5000, top_n: int = 10):
    """Run trio exotic backtest. Delegates to run_sprint5.run_step_5_5."""
    banner("Step 5 — Trio (三連複) Exotic Backtest (Hypothesis H4)")
    from models.run_sprint5 import run_step_5_5
    return run_step_5_5(budget=budget, top_n=top_n)


# ---------------------------------------------------------------------------
# Decision Matrix
# ---------------------------------------------------------------------------

def print_decision_matrix(
    data_counts: Optional[dict] = None,
    ev_results: Optional[list] = None,
    odds_free_results: Optional[list] = None,
    trio_result=None,
    failure_modes: Optional[dict] = None,
):
    """Print the decision matrix from the post-scrape plan."""
    banner("Decision Matrix")

    rows = []

    # Check 1: ROI at 10% EV threshold
    if ev_results:
        for ev_thresh, result in ev_results:
            if abs(ev_thresh - 0.10) < 0.01:
                roi_10 = result.roi_pct
                status = "✅" if roi_10 > 0 else "❌"
                action = (
                    "Lower live threshold 12% → 10%"
                    if roi_10 > 0
                    else "Keep 12%, focus on more data (2019–2021)"
                )
                rows.append((f"ROI at 10% EV: {roi_10:+.1f}%", status, action))

    # Check 2: Odds-free AUC
    if odds_free_results:
        # AUC is reported per-run; we'd need to capture it separately
        # For now, just check if any profitable threshold exists
        rows.append(("Odds-free evaluation", "✅ Run", "Check AUC in output above"))

    # Check 3: Trio ROI vs win ROI
    if trio_result is not None:
        trio_roi = getattr(trio_result, "roi_pct", None)
        if trio_roi is not None:
            status = "✅" if trio_roi > 0 else "❌"
            action = (
                "Add trio to paper trading portfolio"
                if trio_roi > 0
                else "Stick with win bets only"
            )
            rows.append((f"Trio ROI: {trio_roi:+.1f}%", status, action))

    # Check 4: Dominant failure mode
    if failure_modes:
        dominant = max(failure_modes.values(), key=lambda m: m.count)
        pct = dominant.count / sum(m.count for m in failure_modes.values()) * 100
        if pct > 50:
            rows.append((
                f"Dominant failure: {dominant.name} ({pct:.0f}%)",
                "⚠️",
                f"Target {dominant.name} in next feature sprint",
            ))
        else:
            rows.append(("No dominant failure mode", "✅", "Failures spread evenly"))

    # Print
    print(f"  {'Check':<35} {'Status':>6}  Action")
    print("  " + "-" * 70)
    for check, status, action in rows:
        print(f"  {check:<35} {status:>6}  {action}")
    print("  " + "-" * 70)

    if not rows:
        print("  (No results to display — run all steps first)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UmaEdge — Post-Scrape Evaluation Runner"
    )
    parser.add_argument(
        "--only", type=int, choices=[1, 2, 3, 4, 5],
        help="Run only step N (1=data, 2=EV sweep, 3=odds-free, 4=losers, 5=trio)",
    )
    parser.add_argument(
        "--skip-trio", action="store_true",
        help="Skip the trio exotic backtest (step 5)",
    )
    parser.add_argument(
        "--budget", type=int, default=5000,
        help="Trio budget per race in yen (default: 5000)",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Trio max tickets per race (default: 10)",
    )
    parser.add_argument(
        "--ev-threshold", type=float, default=0.05,
        help="EV threshold for losing bet analysis (default: 0.05)",
    )
    args = parser.parse_args()

    start = datetime.now()
    banner(f"Post-Scrape Evaluation — {start.strftime('%Y-%m-%d %H:%M')}")

    data_counts = None
    ev_results = None
    odds_free_results = None
    failure_modes = None
    trio_result = None

    if args.only:
        if args.only == 1:
            data_counts = step_verify_data()
        elif args.only == 2:
            ev_results = step_ev_sweep()
        elif args.only == 3:
            odds_free_results = step_odds_free()
        elif args.only == 4:
            failure_modes = step_losing_bets(args.ev_threshold)
        elif args.only == 5:
            trio_result = step_trio_backtest(args.budget, args.top_n)
    else:
        # Run all steps
        data_counts = step_verify_data()
        ev_results = step_ev_sweep()
        odds_free_results = step_odds_free()
        failure_modes = step_losing_bets(args.ev_threshold)

        if not args.skip_trio:
            trio_result = step_trio_backtest(args.budget, args.top_n)

    # Print decision matrix
    print_decision_matrix(
        data_counts=data_counts,
        ev_results=ev_results,
        odds_free_results=odds_free_results,
        trio_result=trio_result,
        failure_modes=failure_modes,
    )

    elapsed = datetime.now() - start
    banner(f"Evaluation Complete — {elapsed.total_seconds() / 60:.1f} min")


if __name__ == "__main__":
    main()
