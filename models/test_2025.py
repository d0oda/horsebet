"""
UmaEdge — 2025 Model Evaluation.

Trains the LightGBM + XGBoost ensemble on pre-2025 data,
validates on 2025 races, and runs a full backtesting simulation.

Usage:
    python -m models.test_2025
    python -m models.test_2025 --ev-threshold 0.08 --output data/backtest_2025_results.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_2025")


# ---------------------------------------------------------------------------
# Data Stats — quick sanity check of what's in the DB
# ---------------------------------------------------------------------------

def print_db_stats():
    """Print a summary of race data in the database."""
    from scraper.db import get_session
    from sqlalchemy import text

    with get_session() as session:
        # Race counts by year
        rows = session.execute(text("""
            SELECT
                EXTRACT(YEAR FROM date) AS year,
                COUNT(*) AS races,
                COUNT(DISTINCT course_id) AS venues,
                MIN(date) AS earliest,
                MAX(date) AS latest
            FROM races
            WHERE field_size IS NOT NULL
            GROUP BY EXTRACT(YEAR FROM date)
            ORDER BY year
        """)).fetchall()

        print("\n" + "=" * 65)
        print("  UmaEdge — Database Summary")
        print("=" * 65)
        print(f"  {'Year':<6} {'Races':>6} {'Venues':>7} {'Earliest':>12} {'Latest':>12}")
        print("  " + "-" * 55)

        total_races = 0
        for row in rows:
            year, races, venues, earliest, latest = row
            total_races += races
            print(f"  {int(year):<6} {races:>6} {venues:>7} {str(earliest):>12} {str(latest):>12}")

        print("  " + "-" * 55)
        print(f"  {'Total':<6} {total_races:>6}")

        # Entry + result counts
        entries = session.execute(text("SELECT COUNT(*) FROM entries")).scalar()
        results = session.execute(text("SELECT COUNT(*) FROM results")).scalar()
        odds_filled = session.execute(
            text("SELECT COUNT(*) FROM entries WHERE odds_win IS NOT NULL AND odds_win > 0")
        ).scalar()

        print(f"\n  Entries:     {entries:>6}")
        print(f"  Results:     {results:>6}")
        print(f"  w/ odds:     {odds_filled:>6}")
        print("=" * 65)

        return total_races


# ---------------------------------------------------------------------------
# Main Evaluation Pipeline
# ---------------------------------------------------------------------------

def run_2025_evaluation(ev_threshold: float = 0.05, output_path: str = None):
    """
    Full pipeline:
      1. Build features for ALL data
      2. Train on pre-2025, validate on 2025
      3. Evaluate model quality
      4. Run backtest simulation on 2025 races
    """
    from models.features import FeatureBuilder
    from models.train import (
        prepare_data,
        train_lightgbm,
        train_xgboost,
        ensemble_predict,
        evaluate_ensemble,
        save_model,
    )
    from models.backtest import Backtester, BacktestConfig

    # --- Step 0: DB stats ---
    print_db_stats()

    # --- Step 1: Build features ---
    log.info("\n📊 Step 1/4 — Building features for all races...")
    fb = FeatureBuilder()
    df = fb.build_features_all()

    if df.empty:
        log.error("No data available. Run the scraper first.")
        return

    # Check we have both 2024 and 2025 data
    if "date" in df.columns:
        years = df["date"].str[:4].unique()
        log.info(f"Years in feature data: {sorted(years)}")
        if "2025" not in years:
            log.error("No 2025 data found! Run: python -m scraper.batch_scrape --year 2025 --max-races 100")
            return

    # --- Step 2: Train on ≤2024, validate on 2025 ---
    log.info("\n🏋️ Step 2/4 — Training model (train ≤2024, val ≥2025)...")
    X_train, y_train, X_val, y_val, feature_cols, race_ids_val = prepare_data(
        df, target="target_win", val_date="2025-01-01"
    )

    if len(X_train) < 50:
        log.error(f"Not enough training data ({len(X_train)} entries)")
        return
    if len(X_val) < 10:
        log.error(f"Not enough validation data ({len(X_val)} entries)")
        return

    # Train LightGBM
    log.info("\n--- Training LightGBM ---")
    lgb_model, lgb_preds = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols)

    # Train XGBoost
    log.info("\n--- Training XGBoost ---")
    xgb_model, xgb_preds = train_xgboost(X_train, y_train, X_val, y_val, feature_cols)

    # Ensemble
    log.info("\n--- Ensemble Evaluation ---")
    ensemble_preds = ensemble_predict(lgb_preds, xgb_preds)
    metrics = evaluate_ensemble(y_val, ensemble_preds, label="2025 Holdout")

    # Save model (try default dir, fall back to /tmp)
    try:
        version = save_model(lgb_model, xgb_model, feature_cols, metrics, version="2025_test")
    except PermissionError:
        import tempfile
        from pathlib import Path as TmpPath
        tmp_model_dir = TmpPath(tempfile.gettempdir()) / "umaedge_models" / "2025_test"
        tmp_model_dir.mkdir(parents=True, exist_ok=True)
        lgb_model.save_model(str(tmp_model_dir / "lgb_model.txt"))
        xgb_model.save_model(str(tmp_model_dir / "xgb_model.json"))
        import json as _json
        with open(tmp_model_dir / "metadata.json", "w") as f:
            _json.dump({"version": "2025_test", "feature_cols": feature_cols, "metrics": metrics}, f, indent=2)
        version = "2025_test"
        log.info(f"💾 Models saved to {tmp_model_dir} (fallback due to permissions)")
    log.info(f"Model saved as version: {version}")

    # --- Step 3: Build prediction dataframe for backtest ---
    log.info("\n📈 Step 3/4 — Building prediction dataframe for backtest...")

    # Get the 2025 subset from the features dataframe
    val_df = df[df["date"] >= "2025-01-01"].copy()

    # Generate predictions using the trained model
    import xgboost as xgb

    X_2025 = val_df[feature_cols].fillna(0).values
    lgb_probs = lgb_model.predict(X_2025)
    xgb_probs = xgb_model.predict(xgb.DMatrix(X_2025, feature_names=feature_cols))
    val_df["win_prob"] = ensemble_predict(lgb_probs, xgb_probs)

    # We need odds_win and finish_pos for the backtest
    pred_df = val_df[["race_id", "entry_id", "win_prob", "date"]].copy()

    # Merge with odds and results from DB
    from scraper.db import get_session
    from sqlalchemy import text as sql_text

    with get_session() as session:
        data = session.execute(sql_text("""
            SELECT
                e.id AS entry_id,
                e.odds_win,
                h.name_jp AS horse_name,
                r.race_name_jp AS race_name,
                res.finish_pos
            FROM entries e
            JOIN races r ON r.id = e.race_id
            JOIN horses h ON h.id = e.horse_id
            LEFT JOIN results res ON res.entry_id = e.id
            WHERE r.date >= '2025-01-01'
        """)).fetchall()

    extra_df = pd.DataFrame(
        data, columns=["entry_id", "odds_win", "horse_name", "race_name", "finish_pos"]
    )
    pred_df = pred_df.merge(extra_df, on="entry_id", how="left")

    log.info(f"Prediction dataframe: {len(pred_df)} entries, "
             f"{pred_df['race_id'].nunique()} races")
    log.info(f"Entries with odds: {pred_df['odds_win'].notna().sum()}")
    log.info(f"Entries with results: {pred_df['finish_pos'].notna().sum()}")

    # --- Step 4: Run backtest ---
    log.info("\n💰 Step 4/4 — Running backtest on 2025 races...")
    config = BacktestConfig(ev_threshold=ev_threshold)
    bt = Backtester(config)
    result = bt.run(pred_df)

    # Print full report
    bt.print_report(result)

    # --- Additional analysis ---
    print("\n" + "=" * 60)
    print("  2025 Model Performance Summary")
    print("=" * 60)
    print(f"  Model metrics on 2025 holdout:")
    print(f"    Log-loss:  {metrics['logloss']:.4f}")
    print(f"    AUC:       {metrics['auc']:.4f}")
    print(f"    Brier:     {metrics['brier']:.4f}")
    print("-" * 60)
    print(f"  Backtest results (EV threshold: {ev_threshold:.0%}):")
    print(f"    Total bets:  {result.total_bets}")
    print(f"    Hit rate:    {result.hit_rate:.1f}%")
    print(f"    ROI:         {result.roi_pct:+.1f}%")
    print(f"    Sharpe:      {result.sharpe:.2f}")
    print(f"    Max DD:      ¥{result.max_drawdown:,} ({result.max_drawdown_pct:.1f}%)")
    print("=" * 60)

    # --- Top value bets found ---
    if result.bets:
        print(f"\n  Top 10 highest-EV bets found:")
        sorted_bets = sorted(result.bets, key=lambda b: b.ev, reverse=True)[:10]
        print(f"  {'Date':<12} {'Horse':<16} {'P(win)':<8} {'Odds':>5} {'EV':>6} {'Result':>8}")
        print("  " + "-" * 60)
        for b in sorted_bets:
            outcome = f"+¥{b.payout}" if b.profit > 0 else f"-¥{b.stake}"
            marker = "✅" if b.profit > 0 else "❌"
            print(
                f"  {b.date:<12} {b.horse_name:<16} "
                f"{b.model_prob:>5.1%}  {b.odds:>5.1f} {b.ev:>+5.2f}  {outcome:>7} {marker}"
            )

    # --- Monthly breakdown ---
    if result.daily_pnl:
        monthly = {}
        for date_str, pnl in result.daily_pnl:
            month = date_str[:7]  # YYYY-MM
            monthly[month] = monthly.get(month, 0) + pnl

        print(f"\n  Monthly P&L:")
        print(f"  {'Month':<10} {'P&L':>10}")
        print("  " + "-" * 22)
        for month, pnl in sorted(monthly.items()):
            marker = "📈" if pnl >= 0 else "📉"
            print(f"  {month:<10} ¥{pnl:>+9,} {marker}")

    # --- Save CSV ---
    if output_path is None:
        output_path = "data/backtest_2025_results.csv"

    out_dir = Path(output_path).parent
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # test write
        test_file = out_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (PermissionError, OSError):
        import tempfile
        output_path = str(Path(tempfile.gettempdir()) / "backtest_2025_results.csv")

    report_df = bt.to_dataframe(result)
    if not report_df.empty:
        report_df.to_csv(output_path, index=False)
        log.info(f"\n📄 Results saved to {output_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — 2025 Model Evaluation")
    parser.add_argument(
        "--ev-threshold", type=float, default=0.05,
        help="Minimum EV to trigger a bet (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: data/backtest_2025_results.csv)"
    )
    args = parser.parse_args()

    run_2025_evaluation(
        ev_threshold=args.ev_threshold,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
