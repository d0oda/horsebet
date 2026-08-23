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

from models.features import FeatureBuilder, ODDS_FEATURES
from models.train import (
    prepare_data, train_lightgbm, train_xgboost, ensemble_predict,
    train_lightgbm_ranker, train_xgboost_ranker,
    _make_group_array, _finish_to_relevance, scores_to_probs,
)
from models.backtest import Backtester, BacktestConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("walk_forward_backtest")

def run_walk_forward_backtest(n_folds=5, ev_threshold=0.10, use_cache=False, use_kelly=True, flat_stake=1000, initial_bankroll=100000, kelly_fraction=0.25, max_odds=30.0, min_odds=2.0, odds_free=False, use_ranker=False, snapshot_only=False, exotic=False):
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
    
    if snapshot_only:
        log.info("Filtering dataset to ONLY include races with odds_drift data...")
        before = len(df)
        df = df[df["odds_drift"].notna()]
        log.info(f"Filtered {before} -> {len(df)} entries.")

    target = "target_win"
    exclude = {"race_id", "entry_id", "target_win", "target_place", "target_margin", "finish_pos", "date", "horse_name"}
    # Include both numeric AND category-typed columns so the backtest trains the
    # same feature set as the final model.  Previously, category columns
    # (sire_id, broodmare_sire_id, going_code, surface_code, draw) were silently
    # dropped because the dtype filter only passed float/int dtypes.
    CATEGORICAL_COLS = {"sire_id", "broodmare_sire_id", "going_code", "surface_code", "draw"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude
        and (
            df[c].dtype in [np.float64, np.float32, np.int64, float, int]
            or (c in CATEGORICAL_COLS and c in df.columns)
        )
    ]

    # Convert expected categoricals to category dtype so LightGBM/XGBoost handle them correctly
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).astype("category")

    valid_cols = [c for c in feature_cols if not df[c].isna().all()]
    feature_cols = valid_cols

    # Odds-free mode: exclude all odds-derived features from MODEL training.
    # The model must predict win probability from fundamentals only.
    # Note: odds_win is still used by the backtester to calculate EV and sizing.
    if odds_free:
        odds_set = set(ODDS_FEATURES)
        # Also exclude z-scored variants
        odds_z = {f"{f}_z" for f in ODDS_FEATURES}
        remove = odds_set | odds_z
        before = len(feature_cols)
        feature_cols = [c for c in feature_cols if c not in remove]
        log.info(f"ODDS-FREE mode: excluded {before - len(feature_cols)} odds features, {len(feature_cols)} features remain")

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

        # Sort by race_id for group-based ranking models
        train_df = train_df.sort_values("race_id")
        val_df = val_df.sort_values("race_id")
        test_df = test_df.sort_values("race_id")

        X_train = train_df[feature_cols].values
        y_train = train_df[target].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target].values
        X_test = test_df[feature_cols].values

        if len(X_val) < 10 or len(X_train) < 50 or len(X_test) == 0:
            continue

        if use_ranker:
            # --- LambdaRank path ---
            # Convert binary win labels to graded relevance from finish_pos
            train_rel = _finish_to_relevance(train_df["finish_pos"].values)
            val_rel = _finish_to_relevance(val_df["finish_pos"].values)
            
            train_groups = _make_group_array(train_df["race_id"].values)
            val_groups = _make_group_array(val_df["race_id"].values)
            
            lgb_model, _ = train_lightgbm_ranker(
                X_train, train_rel, X_val, val_rel, feature_cols,
                train_groups, val_groups,
            )
            xgb_model, _ = train_xgboost_ranker(
                X_train, train_rel, X_val, val_rel, feature_cols,
                train_groups, val_groups,
            )
            
            # Get raw scores on val set to fit calibrator (NOT training set — training
            # predictions are overfit and fitting on them causes leakage).
            import xgboost as xgb_lib
            lgb_scores_val = lgb_model.predict(X_val)
            xgb_scores_val = xgb_model.predict(xgb_lib.DMatrix(X_val, feature_names=feature_cols))
            ensemble_scores_val = ensemble_predict(lgb_scores_val, xgb_scores_val)
            
            # Convert val scores to per-race softmax probs
            ensemble_probs_val = scores_to_probs(ensemble_scores_val, val_df["race_id"].values)
            
            # Fit isotonic calibrator using val labels against val predictions
            from sklearn.isotonic import IsotonicRegression
            calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            calibrator.fit(ensemble_probs_val, y_val)

            # Raw ranking scores on test set
            lgb_scores = lgb_model.predict(X_test)
            xgb_scores = xgb_model.predict(xgb_lib.DMatrix(X_test, feature_names=feature_cols))
            ensemble_scores = ensemble_predict(lgb_scores, xgb_scores)
            
            # Convert test scores → per-race probabilities via softmax
            ensemble_probs_test = scores_to_probs(ensemble_scores, test_df["race_id"].values)
            
            # Calibrate probabilities
            ensemble_preds = calibrator.predict(ensemble_probs_test)
        else:
            # --- Standard binary classification path ---
            lgb_model, _ = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols)
            xgb_model, _ = train_xgboost(X_train, y_train, X_val, y_val, feature_cols)
            
            lgb_preds = lgb_model.predict(X_test)
            import xgboost as xgb_lib
            xgb_preds = xgb_model.predict(xgb_lib.DMatrix(X_test, feature_names=feature_cols))
            ensemble_preds = ensemble_predict(lgb_preds, xgb_preds)

        # Build prediction dataframe for backtest
        pred_df = test_df[["race_id", "entry_id", "date", "horse_name", "finish_pos"]].copy()
        pred_df["win_prob"] = ensemble_preds

        # Normalise ranker/softmax predictions per race so they sum to 1.0.
        # Binary-classifier calibrated probabilities are absolute estimates and must NOT
        # be normalised — doing so distorts them when the field size causes them to
        # sum to values other than 1.0.  Only normalise for the ranker path.
        if use_ranker:
            pred_df["win_prob"] = pred_df.groupby("race_id")["win_prob"].transform(
                lambda x: x / x.sum() if x.sum() > 0 else x
            )

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
    
    # Save OOS predictions for fast parameter sweeping
    final_pred_df.to_parquet("data/oos_preds.parquet")
    log.info("Saved out-of-sample predictions to data/oos_preds.parquet for sweeping.")
    
    if exotic:
        from models.exotic_backtester import ExoticBacktester
        log.info("\n=== Exotic Walk-Forward Backtest Simulation ===")
        bt = ExoticBacktester(final_pred_df)
        result = bt.run()
        return result
    else:
        log.info("\n=== Walk-Forward Backtest Simulation Results ===")
        config = BacktestConfig(
            ev_threshold=ev_threshold, 
            bet_type="win",
            use_kelly=use_kelly,
            flat_stake=flat_stake,
            initial_bankroll=initial_bankroll,
            kelly_fraction=kelly_fraction,
            max_odds=max_odds,
            min_odds=min_odds,
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
    parser.add_argument("--max-odds", type=float, default=30.0, help="Max odds to bet on")
    parser.add_argument("--min-odds", type=float, default=2.0, help="Min odds to bet on")
    parser.add_argument("--odds-free", action="store_true", help="Exclude odds features from model training (fundamental-only)")
    parser.add_argument("--ranker", action="store_true", help="Use LambdaRank objective instead of binary classification")
    parser.add_argument("--snapshot-only", action="store_true", help="Only evaluate on races that have odds snapshot data")
    parser.add_argument("--exotic", action="store_true", help="Run the Best Bet Exotic Pool recommender backtest instead of Win pool")
    args = parser.parse_args()
    
    run_walk_forward_backtest(
        n_folds=args.n_folds, 
        ev_threshold=args.ev_threshold, 
        use_cache=args.use_cache,
        use_kelly=not args.flat_bet,
        flat_stake=args.flat_stake,
        initial_bankroll=args.initial_bankroll,
        kelly_fraction=args.kelly_fraction,
        max_odds=args.max_odds,
        min_odds=args.min_odds,
        odds_free=args.odds_free,
        use_ranker=args.ranker,
        snapshot_only=args.snapshot_only,
        exotic=args.exotic,
    )
