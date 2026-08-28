"""
UmaEdge — Feature Population Diagnostic.

Checks whether the 6 new feature groups added during retraining
actually contain data or are mostly NaN/zero. This helps diagnose
why the retrained model yields 0 bets.

Usage:
    python -m models.verify_features
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("verify_features")

# New feature groups to check
FEATURE_GROUPS = {
    "Class Change": [
        "class_change", "class_drops_last5", "class_rises_last5", "class_at_last_win",
    ],
    "Trainer 14-Day Form": [
        "trainer_14d_runs", "trainer_14d_win_pct", "trainer_14d_place_pct",
    ],
    "Course × Jockey": [
        "jockey_course_runs", "jockey_course_win_pct", "jockey_course_place_pct",
    ],
    "Weather × Surface": [
        "weather_code", "going_x_surface", "going_x_distance",
        "horse_going_win_pct", "horse_wet_track_advantage", "horse_heavy_speed_diff",
        "going_x_dist_x_surface",
    ],

    "Odds (Cross-Sectional)": [
        "odds_rank", "odds_ratio_to_fav", "odds_deviation",
    ],
    "Track Bias": [
        "draw_bias_at_course", "draw_low_win_pct", "draw_high_win_pct",
        "draw_bias_score", "course_month_bias",
    ],
    "Pedigree (Sire)": [
        "sire_runners", "sire_win_pct", "sire_win_pct_surface",
        "sire_win_pct_distance", "sire_avg_finish",
    ],
    "Pedigree Fallback (Trainer)": [
        "trainer_offspring_win_pct", "trainer_offspring_avg_finish",
    ],
    "Speed Figures": [
        "speed_figure_last", "speed_figure_best", "speed_figure_avg3",
    ],
    "Jockey-Trainer Combo": [
        "jt_combo_runs", "jt_combo_win_pct", "jt_combo_place_pct",
    ],
    "Beaten Lengths": [
        "beaten_lengths_avg3", "beaten_lengths_best", "class_adjusted_margin",
    ],
    "Fitness Curve": [
        "is_fresh", "is_rested", "is_stale",
    ],
    "Age × Class": [
        "age_x_class", "is_improving_3yo",
    ],
    "Field Quality": [
        "field_avg_career_win_pct", "horse_vs_field_quality",
    ],
    "Weight vs Field": [
        "weight_vs_field_avg", "weight_per_kg_body",
    ],
}


def main():
    from models.features import FeatureBuilder

    log.info("📊 Building features for all races...")
    fb = FeatureBuilder()
    df = fb.build_features_all()

    if df.empty:
        log.error("No data available. Run the scraper first.")
        return

    n_rows = len(df)
    log.info(f"Total entries: {n_rows}")
    log.info(f"Total columns: {len(df.columns)}")
    log.info(f"Date range: {df['date'].min()} to {df['date'].max()}" if "date" in df.columns else "")

    print("\n" + "=" * 80)
    print("  Feature Population Diagnostic")
    print("=" * 80)

    all_results = []

    for group_name, feature_names in FEATURE_GROUPS.items():
        print(f"\n--- {group_name} ---")
        found = [f for f in feature_names if f in df.columns]
        missing = [f for f in feature_names if f not in df.columns]

        if missing:
            print(f"  ⚠️  Missing from dataframe: {missing}")

        if not found:
            print(f"  ❌ NONE of the expected features exist in the dataframe!")
            for f in feature_names:
                all_results.append({
                    "group": group_name, "feature": f,
                    "status": "MISSING", "nan_pct": 100.0, "zero_pct": 0.0,
                    "non_null_count": 0, "mean": None, "std": None,
                })
            continue

        for feat in found:
            col = df[feat]
            nan_count = col.isna().sum()
            nan_pct = nan_count / n_rows * 100
            non_null = col.dropna()
            zero_count = (non_null == 0).sum() if len(non_null) > 0 else 0
            zero_pct = zero_count / n_rows * 100
            populated_pct = 100 - nan_pct

            # Determine status
            if nan_pct >= 99:
                status = "❌ EMPTY"
            elif nan_pct >= 80:
                status = "⚠️  SPARSE"
            elif nan_pct >= 50:
                status = "🟡 PARTIAL"
            else:
                status = "✅ GOOD"

            mean_val = non_null.mean() if len(non_null) > 0 else None
            std_val = non_null.std() if len(non_null) > 0 else None

            print(
                f"  {status:14s} {feat:30s}  "
                f"populated={populated_pct:5.1f}%  "
                f"zeros={zero_pct:5.1f}%  "
                f"mean={mean_val:+8.4f}" if mean_val is not None else
                f"  {status:14s} {feat:30s}  "
                f"populated={populated_pct:5.1f}%  "
                f"zeros={zero_pct:5.1f}%  "
                f"mean=     N/A"
            )

            all_results.append({
                "group": group_name, "feature": feat,
                "status": status.strip(), "nan_pct": nan_pct, "zero_pct": zero_pct,
                "non_null_count": len(non_null),
                "mean": float(mean_val) if mean_val is not None else None,
                "std": float(std_val) if std_val is not None else None,
            })

    # Summary table
    print("\n" + "=" * 80)
    print("  Summary by Group")
    print("=" * 80)
    print(f"  {'Group':30s}  {'Avg Populated %':>15s}  {'Verdict':>10s}")
    print("  " + "-" * 60)

    for group_name in FEATURE_GROUPS:
        group_results = [r for r in all_results if r["group"] == group_name]
        avg_populated = 100 - np.mean([r["nan_pct"] for r in group_results])
        if avg_populated >= 50:
            verdict = "✅ OK"
        elif avg_populated >= 10:
            verdict = "⚠️  LOW"
        else:
            verdict = "❌ DEAD"
        print(f"  {group_name:30s}  {avg_populated:14.1f}%  {verdict:>10s}")

    # Also check z-scored variants of new features
    print("\n--- Z-scored variants of new features ---")
    z_cols = [c for c in df.columns if c.endswith("_z") and any(
        base in c for group in FEATURE_GROUPS.values() for base in group
    )]
    if z_cols:
        for col_name in z_cols:
            col = df[col_name]
            nan_pct = col.isna().sum() / n_rows * 100
            print(f"  {col_name:35s}  populated={100 - nan_pct:5.1f}%")
    else:
        print("  (none found)")

    print("\n" + "=" * 80)
    print("  Done. Use this report to decide which features to keep/drop.")
    print("=" * 80)


if __name__ == "__main__":
    main()
