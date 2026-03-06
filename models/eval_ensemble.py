#!/usr/bin/env python3
"""
UmaEdge — Ensemble vs Single Model Evaluation.

Compares:
  1. LightGBM only (market)
  2. XGBoost only (market)
  3. LGB+XGB ensemble (market)
  4. HybridEnsemble (fundamental + market adaptive blend)

Uses cached features to avoid the 60-min rebuild.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_ensemble")

FEATURES_CACHE = os.path.join(os.path.dirname(__file__), "saved", "features_cache.pkl")


def evaluate(y_true, y_pred, label):
    """Compute key metrics."""
    ll = log_loss(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    return {"label": label, "logloss": ll, "auc": auc, "brier": brier}


def main():
    from models.features import FeatureBuilder, ODDS_FEATURES
    from models.train import (
        prepare_data, train_lightgbm, train_xgboost,
        ensemble_predict, calibrate_predictions,
    )
    from models.ensemble import HybridEnsemble
    from models.backtest import Backtester, BacktestConfig

    t0 = time.time()

    # 1. Load cached features
    if not os.path.exists(FEATURES_CACHE):
        log.error(f"No cached features at {FEATURES_CACHE}. Run multiday_backtest --rebuild first.")
        sys.exit(1)

    log.info("Loading cached features...")
    df = pd.read_pickle(FEATURES_CACHE)
    log.info(f"Loaded {len(df)} entries with {len(df.columns)} columns")

    # Split train / test (pre-2025 / 2025+)
    df["year"] = df["date"].str[:4].astype(float)
    train_df = df[df["year"] < 2025].copy()
    test_df = df[df["year"] >= 2025].copy()
    log.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

    results = []

    # ================================================================
    # Model A: LightGBM only (market, with odds)
    # ================================================================
    log.info("\n" + "=" * 60)
    log.info("  MODEL A: LightGBM only (market)")
    log.info("=" * 60)

    X_train, y_train, X_val, y_val, feat_cols, _ = prepare_data(
        train_df.copy(), target="target_win",
    )

    lgb_model, lgb_val_preds = train_lightgbm(X_train, y_train, X_val, y_val, feat_cols)
    results.append(evaluate(y_val, lgb_val_preds, "LightGBM only"))

    # ================================================================
    # Model B: XGBoost only (market, with odds)
    # ================================================================
    log.info("\n" + "=" * 60)
    log.info("  MODEL B: XGBoost only (market)")
    log.info("=" * 60)

    xgb_model, xgb_val_preds = train_xgboost(X_train, y_train, X_val, y_val, feat_cols)
    results.append(evaluate(y_val, xgb_val_preds, "XGBoost only"))

    # ================================================================
    # Model C: LGB+XGB Ensemble (simple average, market)
    # ================================================================
    log.info("\n" + "=" * 60)
    log.info("  MODEL C: LGB+XGB Ensemble (market)")
    log.info("=" * 60)

    ens_preds = ensemble_predict(lgb_val_preds, xgb_val_preds)
    results.append(evaluate(y_val, ens_preds, "LGB+XGB Ensemble"))

    # ================================================================
    # Model D: LGB+XGB Ensemble + Isotonic Calibration
    # ================================================================
    log.info("\n" + "=" * 60)
    log.info("  MODEL D: LGB+XGB + Isotonic Calibration")
    log.info("=" * 60)

    lgb_train_preds = lgb_model.predict(X_train)
    import xgboost as xgb_lib
    xgb_train_preds = xgb_model.predict(
        xgb_lib.DMatrix(X_train, feature_names=feat_cols)
    )
    ens_train_preds = ensemble_predict(lgb_train_preds, xgb_train_preds)
    cal_preds, _ = calibrate_predictions(
        y_train, ens_train_preds, y_val, ens_preds, method="isotonic"
    )
    results.append(evaluate(y_val, cal_preds, "LGB+XGB + Isotonic"))

    # ================================================================
    # Model E: HybridEnsemble (fundamental + market adaptive blend)
    # ================================================================
    log.info("\n" + "=" * 60)
    log.info("  MODEL E: HybridEnsemble (fund + market)")
    log.info("=" * 60)

    hybrid = HybridEnsemble(calibration_method="isotonic")
    hybrid.train(train_df.copy())

    # Predict on test set
    preds_dict = hybrid.predict(test_df)

    # For fair comparison, we need y_test
    target_col = "target_win"
    y_test = test_df[target_col].values

    results.append(evaluate(y_test, preds_dict["fundamental"], "Hybrid: Fundamental"))
    results.append(evaluate(y_test, preds_dict["market"], "Hybrid: Market"))
    results.append(evaluate(y_test, preds_dict["combined"], "Hybrid: Combined"))

    # ================================================================
    # Results Summary
    # ================================================================
    print("\n" + "=" * 75)
    print("  ENSEMBLE vs SINGLE MODEL COMPARISON")
    print("=" * 75)
    print(f"{'Model':<30} {'LogLoss':>10} {'AUC':>10} {'Brier':>10}")
    print("-" * 75)

    for r in results:
        print(f"  {r['label']:<28} {r['logloss']:>10.4f} {r['auc']:>10.4f} {r['brier']:>10.4f}")

    print("-" * 75)

    # Find best
    best_auc = max(results, key=lambda x: x["auc"])
    best_ll = min(results, key=lambda x: x["logloss"])
    print(f"\n  Best AUC:     {best_auc['label']} ({best_auc['auc']:.4f})")
    print(f"  Best LogLoss: {best_ll['label']} ({best_ll['logloss']:.4f})")

    # Ensemble uplift
    lgb_auc = results[0]["auc"]
    xgb_auc = results[1]["auc"]
    ens_auc = results[2]["auc"]
    avg_single = (lgb_auc + xgb_auc) / 2
    print(f"\n  Ensemble uplift vs avg single: {(ens_auc - avg_single)*100:+.2f}pp AUC")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
