"""
UmaEdge — Sprint 5 Consolidated Runner.

Runs all Sprint 5 evaluation steps in sequence:
  5.2 — EV sweep with expanded data (isotonic calibration)
  5.3 — Odds-free backtest EV sweep
  5.4 — Losing bet analysis at 5% threshold
  5.5 — Trio exotic market backtest

Usage:
    python -m models.run_sprint5
    python -m models.run_sprint5 --skip-trio   # skip trio backtest
    python -m models.run_sprint5 --only-trio   # run only trio test
"""

import argparse
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sprint5")


def banner(title: str):
    print(f"\n{'#' * 70}")
    print(f"#  {title}")
    print(f"{'#' * 70}\n")


def run_step_5_2():
    """5.2 — Re-run EV sweep with expanded data + isotonic calibration."""
    banner("Sprint 5.2 — EV Sweep (with odds, isotonic calibration)")
    from models.test_2025 import run_ev_sweep
    run_ev_sweep(exclude_odds=False, calibration_method="none")


def run_step_5_3():
    """5.3 — Odds-free backtest EV sweep."""
    banner("Sprint 5.3 — EV Sweep (ODDS-FREE, isotonic calibration)")
    from models.test_2025 import run_ev_sweep
    run_ev_sweep(exclude_odds=True, calibration_method="none")


def run_step_5_4():
    """5.4 — Analyse losing bets at 5% EV threshold."""
    banner("Sprint 5.4 — Losing Bet Analysis (5% EV threshold)")
    from models.test_2025 import run_2025_evaluation
    from models.analyse_bets import analyse_losing_bets

    result = run_2025_evaluation(
        ev_threshold=0.05,
        calibration_method="none",
    )
    if result and result.bets:
        modes = analyse_losing_bets(result)
        return modes
    else:
        log.warning("No bets to analyse at 5% threshold")
        return None


def run_step_5_5(budget: int = 5000, top_n: int = 10):
    """5.5 — Trio exotic market backtest."""
    banner("Sprint 5.5 — Trio (三連複) Exotic Backtest")

    from models.features import FeatureBuilder
    from models.train import prepare_data, train_lightgbm, train_xgboost, ensemble_predict
    from models.backtest_trio import run_trio_backtest, print_trio_report
    import xgboost as xgb

    log.info("Building features...")
    fb = FeatureBuilder()
    df = fb.build_features_all()
    if df.empty:
        log.error("No data — run scraper first")
        return None

    log.info("Preparing data...")
    X_train, y_train, X_val, y_val, feature_cols, _, _ = prepare_data(
        df, target="target_win", val_date="2025-01-01"
    )

    log.info("Training models...")
    lgb_model, _ = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols)
    xgb_model, _ = train_xgboost(X_train, y_train, X_val, y_val, feature_cols)

    # Predict on validation (2025) set
    lgb_val = lgb_model.predict(X_val)
    xgb_val = xgb_model.predict(xgb.DMatrix(X_val, feature_names=feature_cols))
    preds = ensemble_predict(lgb_val, xgb_val)

    val_df = df[df["date"] >= "2025-01-01"].copy()
    val_df["win_prob"] = preds

    # Need horse_id and finish_pos for trio matching
    from scraper.db import get_session
    from sqlalchemy import text as sql_text
    import pandas as pd

    with get_session() as session:
        data = session.execute(sql_text("""
            SELECT
                e.id AS entry_id,
                e.horse_id,
                e.odds_win,
                h.name_jp AS horse_name,
                res.finish_pos
            FROM entries e
            JOIN horses h ON h.id = e.horse_id
            LEFT JOIN results res ON res.entry_id = e.id
            JOIN races r ON r.id = e.race_id
            WHERE r.date >= '2025-01-01'
        """)).fetchall()

    extra_df = pd.DataFrame(
        data, columns=["entry_id", "horse_id", "odds_win", "horse_name", "finish_pos"]
    )
    val_df = val_df.merge(extra_df, on="entry_id", how="left", suffixes=("", "_db"))

    result = run_trio_backtest(
        val_df,
        budget_per_race=budget,
        top_n_tickets=top_n,
    )
    print_trio_report(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Sprint 5 Runner")
    parser.add_argument("--skip-trio", action="store_true", help="Skip trio backtest")
    parser.add_argument("--only-trio", action="store_true", help="Run only trio backtest")
    parser.add_argument("--budget", type=int, default=5000, help="Trio budget/race (yen)")
    parser.add_argument("--top-n", type=int, default=10, help="Trio tickets/race")
    args = parser.parse_args()

    start = datetime.now()
    banner(f"Sprint 5 — Starting at {start.strftime('%Y-%m-%d %H:%M')}")

    # Check DB first
    from models.test_2025 import print_db_stats
    total_races = print_db_stats()

    if args.only_trio:
        trio_result = run_step_5_5(budget=args.budget, top_n=args.top_n)
    else:
        # 5.2 — EV sweep with expanded data
        run_step_5_2()

        # 5.3 — Odds-free EV sweep
        run_step_5_3()

        # 5.4 — Losing bet analysis
        run_step_5_4()

        # 5.5 — Trio exotic backtest
        if not args.skip_trio:
            trio_result = run_step_5_5(budget=args.budget, top_n=args.top_n)

    elapsed = datetime.now() - start
    banner(f"Sprint 5 Complete — {elapsed.total_seconds() / 60:.1f} min")


if __name__ == "__main__":
    main()
