"""
UmaEdge — Walk-Forward Backtest Simulation.

Iteratively trains models chronologically and simulates betting
on the out-of-sample validation periods. Concatenates all
out-of-sample predictions and generates a single continuous
ROI summary curve.

Usage:
    python -m models.walk_forward_backtest --n-folds 5 --ev-threshold 0.10
"""

import argparse
import logging
import os
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

from models.features import FeatureBuilder
from models.train import prepare_data, train_lightgbm, train_xgboost, ensemble_predict
from models.backtest import Backtester, BacktestConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("walk_forward_backtest")

def run_walk_forward_backtest(n_folds=5, ev_threshold=0.10, use_cache=False, use_kelly=True, flat_stake=1000, initial_bankroll=100000, kelly_fraction=0.25):
    cache_path = "data/features.parquet"
    if use_cache and os.path.exists(cache_path):
        log.info(f"Loading cached features from {cache_path}...")
        df = pd.read_parquet(cache_path)
    else:
        log.info("Building features from database ...")
        fb = FeatureBuilder()
        df = fb.build_features_all()
        
        if not df.empty:
            log.info(f"Saving features to cache {cache_path}...")
            os.makedirs("data", exist_ok=True)
            # Ensure date is string before parquet serialization if needed
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
            df.to_parquet(cache_path, index=False)
    
    if df.empty:
        log.error("No data found")
        return

    df = df.dropna(subset=["target_win", "date"]).copy()
    df = df.sort_values("date")

    target = "target_win"
    exclude = {"race_id", "entry_id", "target_win", "target_place", "finish_pos", "date", "horse_name"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, float, int]]
    
    valid_cols = [c for c in feature_cols if not df[c].isna().all()]
    feature_cols = valid_cols

    unique_dates = sorted(df["date"].unique())
    total_dates = len(unique_dates)
    chunk_size = max(1, total_dates // (n_folds + 1))

    all_out_of_sample_preds = []

    for fold_i in range(n_folds):
        test_start_idx = (fold_i + 1) * chunk_size
        test_end_idx = min(test_start_idx + chunk_size, total_dates)

        if test_start_idx >= total_dates:
            break

        test_dates = unique_dates[test_start_idx:test_end_idx]
        if not test_dates:
            break

        # Use the last 50% of the previous chunk as the early-stopping validation set
        val_start_idx = max(0, test_start_idx - int(chunk_size * 0.5))
        val_dates = unique_dates[val_start_idx:test_start_idx]
        train_cutoff = unique_dates[val_start_idx]

        train_df = df[df["date"] < train_cutoff].copy()
        val_df = df[df["date"].isin(val_dates)].copy()
        test_df = df[df["date"].isin(test_dates)].copy()

        n_train_races = train_df["race_id"].nunique()
        log.info(f"Fold {fold_i + 1}: train={n_train_races} races, val={val_df['race_id'].nunique()} races, test={test_df['race_id'].nunique()} races ({test_dates[0]}→{test_dates[-1]})")
        
        # Removed manual NaN imputation here.
        # Tree-based models natively support and optimize missing value splits.

        X_train = train_df[feature_cols].values
        y_train = train_df[target].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target].values
        X_test = test_df[feature_cols].values

        if len(X_val) < 10 or len(X_train) < 50 or len(X_test) == 0:
            continue

        lgb_model, _ = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols)
        xgb_model, _ = train_xgboost(X_train, y_train, X_val, y_val, feature_cols)
        
        # Predict purely out-of-sample on the quarantined test set
        lgb_preds = lgb_model.predict(X_test)
        import xgboost as xgb_lib
        xgb_preds = xgb_model.predict(xgb_lib.DMatrix(X_test, feature_names=feature_cols))
        ensemble_preds = ensemble_predict(lgb_preds, xgb_preds)

        # Build prediction dataframe for backtest
        pred_df = test_df[["race_id", "entry_id", "date", "horse_name", "finish_pos"]].copy()
        pred_df["win_prob"] = ensemble_preds

        # Ensure we have odds available
        if "odds_win" in test_df.columns:
            pred_df["odds_win"] = test_df["odds_win"].values
        else:
            from scraper.db import get_session
            from sqlalchemy import text
            with get_session() as session:
                race_ids_tuple = tuple(pred_df["race_id"].unique())
                if race_ids_tuple:
                    data = session.execute(text(f"""
                        SELECT e.id AS entry_id, e.odds_win 
                        FROM entries e
                        WHERE e.race_id IN :rids
                    """), {"rids": race_ids_tuple}).fetchall()
                    odds_df = pd.DataFrame(data, columns=["entry_id", "odds_win"])
                    pred_df = pred_df.merge(odds_df, on="entry_id", how="left")
                else:
                    pred_df["odds_win"] = 0.0

        all_out_of_sample_preds.append(pred_df)

    if not all_out_of_sample_preds:
        log.warning("No out of sample predictions generated")
        return

    final_pred_df = pd.concat(all_out_of_sample_preds, ignore_index=True)
    
    log.info("\n=== Walk-Forward Backtest Simulation Results ===")
    config = BacktestConfig(
        ev_threshold=ev_threshold, 
        bet_type="win",
        use_kelly=use_kelly,
        flat_stake=flat_stake,
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction
    )
    bt = Backtester(config)
    result = bt.run(final_pred_df)
    bt.print_report(result)
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5, help="Number of cv folds")
    parser.add_argument("--ev-threshold", type=float, default=0.10, help="EV threshold config for backtest")
    parser.add_argument("--use-cache", action="store_true", help="Load features from data/features.parquet if exists")
    parser.add_argument("--flat-bet", action="store_true", help="Use flat betting instead of Kelly criterion")
    parser.add_argument("--flat-stake", type=int, default=1000, help="Stake amount for flat bets (yen)")
    parser.add_argument("--initial-bankroll", type=int, default=100000, help="Starting bankroll (yen)")
    parser.add_argument("--kelly-fraction", type=float, default=0.25, help="Kelly criterion multiplier (e.g. 0.25 = quarter Kelly)")
    args = parser.parse_args()
    
    run_walk_forward_backtest(
        n_folds=args.n_folds, 
        ev_threshold=args.ev_threshold, 
        use_cache=args.use_cache,
        use_kelly=not args.flat_bet,
        flat_stake=args.flat_stake,
        initial_bankroll=args.initial_bankroll,
        kelly_fraction=args.kelly_fraction
    )
