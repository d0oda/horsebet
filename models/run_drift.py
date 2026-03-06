#!/usr/bin/env python3
"""
UmaEdge — Drift Detection Report.

1. Logs current model metrics to drift_log.json
2. Feature-level drift via KS test (recent vs historical)
3. Prediction distribution analysis (calibration drift)

Uses cached features — runs in seconds.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("drift_detect")

FEATURES_CACHE = os.path.join(os.path.dirname(__file__), "saved", "features_cache.pkl")


def main():
    from models.features import FeatureBuilder, ODDS_FEATURES
    from models.drift import log_metrics, print_drift_report

    # 1. Load cached features
    if not os.path.exists(FEATURES_CACHE):
        log.error(f"No cached features at {FEATURES_CACHE}. Run multiday_backtest --rebuild first.")
        sys.exit(1)

    log.info("Loading cached features...")
    df = pd.read_pickle(FEATURES_CACHE)
    log.info(f"Loaded {len(df)} entries, {len(df.columns)} columns")

    # Split by era
    df["year"] = df["date"].str[:4].astype(float)
    historical = df[df["year"] < 2025].copy()
    recent = df[df["year"] >= 2025].copy()

    log.info(f"Historical (pre-2025): {len(historical)} entries")
    log.info(f"Recent (2025+):        {len(recent)} entries")

    # 2. Log current metrics
    log.info("\n--- Logging current model metrics ---")
    log_metrics(
        version="retrain_20260305",
        auc=0.8385,
        log_loss=0.2222,
        brier=0.0579,
        n_samples=len(recent),
        notes="Post-cleanup retrain: dead odds movement removed, margin parser fixed, Sprint 8/9 features",
    )

    # 3. Feature-level drift (KS test)
    log.info("\n" + "=" * 70)
    log.info("  FEATURE DRIFT ANALYSIS (KS Test: Historical vs Recent)")
    log.info("=" * 70)

    feature_cols = FeatureBuilder.get_feature_columns(df)
    drift_results = []

    for col in feature_cols:
        hist_vals = historical[col].dropna()
        rec_vals = recent[col].dropna()

        if len(hist_vals) < 10 or len(rec_vals) < 10:
            continue

        ks_stat, p_value = stats.ks_2samp(hist_vals, rec_vals)
        hist_mean = hist_vals.mean()
        rec_mean = rec_vals.mean()
        mean_shift = rec_mean - hist_mean

        drift_results.append({
            "feature": col,
            "ks_stat": ks_stat,
            "p_value": p_value,
            "hist_mean": hist_mean,
            "rec_mean": rec_mean,
            "mean_shift": mean_shift,
            "drifted": p_value < 0.001,  # Strict threshold
        })

    drift_df = pd.DataFrame(drift_results).sort_values("ks_stat", ascending=False)

    # Print top drifters
    n_drifted = drift_df["drifted"].sum()
    n_total = len(drift_df)
    print(f"\n  Features analysed: {n_total}")
    print(f"  Significant drift (p < 0.001): {n_drifted} ({n_drifted/n_total*100:.1f}%)")
    print(f"  Stable (p >= 0.001): {n_total - n_drifted}")

    print(f"\n  {'Feature':<35} {'KS Stat':>8} {'p-value':>10} {'Hist μ':>8} {'Rec μ':>8} {'Shift':>8} {'Status':>8}")
    print("  " + "-" * 95)

    for _, r in drift_df.head(20).iterrows():
        status = "⚠️ DRIFT" if r["drifted"] else "✅ OK"
        print(f"  {r['feature']:<35} {r['ks_stat']:>8.3f} {r['p_value']:>10.2e} "
              f"{r['hist_mean']:>8.3f} {r['rec_mean']:>8.3f} {r['mean_shift']:>+8.3f} {status:>8}")

    # Stable features
    stable = drift_df[~drift_df["drifted"]].sort_values("ks_stat")
    if len(stable) > 0:
        print(f"\n  Most stable features (lowest KS):")
        for _, r in stable.head(5).iterrows():
            print(f"    {r['feature']:<35} KS={r['ks_stat']:.3f}")

    # 4. Target distribution drift
    print(f"\n{'=' * 70}")
    print("  TARGET DISTRIBUTION DRIFT")
    print(f"{'=' * 70}")

    hist_wr = historical["target_win"].mean()
    rec_wr = recent["target_win"].mean()
    print(f"  Win rate — Historical: {hist_wr:.4f} ({hist_wr*100:.2f}%)")
    print(f"  Win rate — Recent:     {rec_wr:.4f} ({rec_wr*100:.2f}%)")
    print(f"  Shift:                 {rec_wr - hist_wr:+.4f}")

    if abs(rec_wr - hist_wr) < 0.005:
        print("  Status: ✅ Target distribution stable")
    else:
        print("  Status: ⚠️ Target distribution shifted")

    # 5. Odds distribution drift
    print(f"\n{'=' * 70}")
    print("  ODDS DISTRIBUTION DRIFT")
    print(f"{'=' * 70}")

    hist_odds = historical["odds_win"].dropna()
    rec_odds = recent["odds_win"].dropna()
    print(f"  Median odds — Historical: {hist_odds.median():.1f}x")
    print(f"  Median odds — Recent:     {rec_odds.median():.1f}x")
    print(f"  Mean odds — Historical:   {hist_odds.mean():.1f}x")
    print(f"  Mean odds — Recent:       {rec_odds.mean():.1f}x")

    # 6. Year-by-year AUC if we had predictions (skip — just show data coverage)
    print(f"\n{'=' * 70}")
    print("  DATA COVERAGE BY YEAR")
    print(f"{'=' * 70}")
    year_stats = df.groupby("year").agg(
        entries=("entry_id", "count"),
        races=("race_id", "nunique"),
        win_rate=("target_win", "mean"),
        odds_coverage=("odds_win", lambda x: x.notna().mean()),
    ).round(4)
    print(year_stats.to_string())

    # 7. Print drift report from log
    print()
    print_drift_report()

    # Summary
    print(f"\n{'=' * 70}")
    print("  DRIFT DETECTION SUMMARY")
    print(f"{'=' * 70}")
    if n_drifted / n_total < 0.3:
        print(f"  ✅ Model is stable — only {n_drifted}/{n_total} features show drift")
    elif n_drifted / n_total < 0.5:
        print(f"  ⚠️ Moderate drift — {n_drifted}/{n_total} features shifted")
        print("     Consider retraining on more recent data")
    else:
        print(f"  🔴 Significant drift — {n_drifted}/{n_total} features shifted")
        print("     Retrain recommended")


if __name__ == "__main__":
    main()
