"""
UmaEdge — Feature Importance & SHAP Analysis.

Generates feature importance rankings, SHAP summary plots, and
per-prediction explanations. Useful for understanding which features
drive the model and debugging individual predictions.

Usage:
    python -m models.explain                    # feature importance bar chart
    python -m models.explain --shap             # SHAP summary (beeswarm) plot
    python -m models.explain --top 20           # show top 20 features
    python -m models.explain --race 123         # explain predictions for a race
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("explain")

OUTPUT_DIR = Path(__file__).parent / "explanations"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Feature Importance (built-in LightGBM / XGBoost)
# ---------------------------------------------------------------------------

def get_feature_importance(version: str = "latest", top_n: int = 30) -> pd.DataFrame:
    """
    Extract feature importance from saved LightGBM and XGBoost models.
    Returns a DataFrame with columns: feature, lgb_importance, xgb_importance, combined.
    """
    from models.train import load_model

    lgb_model, xgb_model, meta = load_model(version)
    feature_cols = meta.get("feature_cols", meta.get("features", []))

    # LightGBM importance (gain-based)
    lgb_imp = lgb_model.feature_importance(importance_type="gain")
    lgb_imp_norm = lgb_imp / lgb_imp.sum() if lgb_imp.sum() > 0 else lgb_imp

    # XGBoost importance (gain-based)
    xgb_scores = xgb_model.get_score(importance_type="gain")
    xgb_imp = np.array([xgb_scores.get(f, 0) for f in feature_cols])
    xgb_imp_norm = xgb_imp / xgb_imp.sum() if xgb_imp.sum() > 0 else xgb_imp

    # Combined (weighted average matching ensemble weights)
    combined = 0.55 * lgb_imp_norm + 0.45 * xgb_imp_norm

    df = pd.DataFrame({
        "feature": feature_cols,
        "lgb_importance": lgb_imp_norm,
        "xgb_importance": xgb_imp_norm,
        "combined": combined,
    }).sort_values("combined", ascending=False)

    return df.head(top_n).reset_index(drop=True)


def print_importance_table(df: pd.DataFrame):
    """Pretty-print the feature importance table."""
    print(f"\n{'=' * 70}")
    print(f"  {'Feature Importance Rankings (Gain-Based)':^60}")
    print(f"{'=' * 70}")
    print(f"  {'Rank':<6}{'Feature':<35}{'LGB':>8}{'XGB':>8}{'Combined':>10}")
    print(f"  {'-' * 64}")

    for i, row in df.iterrows():
        bar_len = int(row["combined"] * 200)
        bar = "█" * min(bar_len, 20)
        print(
            f"  {i + 1:<6}{row['feature']:<35}"
            f"{row['lgb_importance']:>7.1%}"
            f"{row['xgb_importance']:>8.1%}"
            f"{row['combined']:>9.1%}  {bar}"
        )

    print(f"{'=' * 70}\n")


def save_importance_json(df: pd.DataFrame, filename: str = "feature_importance.json"):
    """Save feature importance to JSON for frontend consumption."""
    out = df.to_dict(orient="records")
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"💾 Saved importance to {path}")
    return path


# ---------------------------------------------------------------------------
# SHAP Analysis
# ---------------------------------------------------------------------------

def run_shap_analysis(
    version: str = "latest",
    n_samples: int = 500,
    top_n: int = 20,
):
    """
    Compute SHAP values for the LightGBM model and generate summary.
    Returns SHAP values DataFrame and saves a text summary.
    """
    try:
        import shap
    except ImportError:
        log.error("SHAP not installed. Run: pip install shap")
        return None

    from models.train import load_model
    from models.features import FeatureBuilder

    log.info("Loading model and building features...")
    lgb_model, xgb_model, meta = load_model(version)
    feature_cols = meta.get("feature_cols", meta.get("features", []))

    fb = FeatureBuilder()
    df = fb.build_features_all()
    if df.empty:
        log.error("No feature data available")
        return None

    # Get feature matrix
    X = df[feature_cols].values
    if len(X) > n_samples:
        idx = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    log.info(f"Computing SHAP values for {len(X_sample)} samples...")
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values may be a list [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    # Mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  {'SHAP Feature Importance (Mean |SHAP|)':^50}")
    print(f"{'=' * 60}")
    for i, row in shap_df.head(top_n).iterrows():
        bar_len = int(row["mean_abs_shap"] / shap_df["mean_abs_shap"].max() * 20)
        bar = "█" * bar_len
        print(f"  {row['feature']:<35}{row['mean_abs_shap']:>8.4f}  {bar}")
    print(f"{'=' * 60}\n")

    # Save
    out_path = OUTPUT_DIR / "shap_importance.json"
    shap_df.head(top_n).to_json(out_path, orient="records", indent=2)
    log.info(f"💾 Saved SHAP importance to {out_path}")

    # Identify Sprint 7 features
    sprint7_features = [
        "weather_code", "going_x_surface", "going_x_distance",
        "horse_going_win_pct", "horse_wet_track_advantage",
        "draw_bias_at_course", "draw_low_win_pct", "draw_high_win_pct",
        "draw_bias_score", "course_month_bias",
        "sire_runners", "sire_win_pct", "sire_win_pct_surface",
        "sire_win_pct_distance", "sire_avg_finish",
    ]
    s7_shap = shap_df[shap_df["feature"].isin(sprint7_features)]
    if not s7_shap.empty:
        print("\n📊 Sprint 7 Feature Rankings:")
        for _, row in s7_shap.iterrows():
            rank = shap_df.index.tolist().index(row.name) + 1
            print(f"  #{rank}: {row['feature']} (SHAP = {row['mean_abs_shap']:.4f})")

    return shap_df


# ---------------------------------------------------------------------------
# Per-Race Explanation
# ---------------------------------------------------------------------------

def explain_race(
    race_id: int,
    version: str = "latest",
    top_n: int = 10,
):
    """
    Explain predictions for all entries in a specific race.
    Shows the top contributing features for each horse.
    """
    try:
        import shap
    except ImportError:
        log.error("SHAP not installed. Run: pip install shap")
        return None

    from models.train import load_model
    from models.features import FeatureBuilder

    lgb_model, xgb_model, meta = load_model(version)
    feature_cols = meta.get("feature_cols", meta.get("features", []))

    fb = FeatureBuilder()
    race_df = fb.build_features_for_race(race_id)
    if race_df.empty:
        log.error(f"No data for race {race_id}")
        return None

    X = race_df[feature_cols].values
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Predictions
    lgb_preds = lgb_model.predict(X)

    print(f"\n{'=' * 70}")
    print(f"  Race {race_id} — Per-Horse Feature Explanations")
    print(f"{'=' * 70}")

    for idx, (_, row) in enumerate(race_df.iterrows()):
        horse_name = row.get("horse_name", f"Horse {idx + 1}")
        pred = lgb_preds[idx]
        sv = shap_values[idx]
        
        # Top contributing features (positive and negative)
        feat_contrib = sorted(
            zip(feature_cols, sv, X[idx]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        print(f"\n  🐴 {horse_name} — P(win) = {pred:.1%}")
        print(f"  {'Feature':<30}{'SHAP':>8}{'Value':>10}")
        print(f"  {'-' * 48}")
        for feat, shap_val, feat_val in feat_contrib[:top_n]:
            direction = "+" if shap_val > 0 else "-"
            print(f"  {feat:<30}{direction}{abs(shap_val):>7.4f}{feat_val:>10.2f}")

    print(f"\n{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Hyperparameter Tuning Recommendations
# ---------------------------------------------------------------------------

def suggest_hyperparameters():
    """
    Print hyperparameter tuning recommendations based on current setup.
    """
    print(f"\n{'=' * 60}")
    print(f"  {'Hyperparameter Tuning Recommendations':^50}")
    print(f"{'=' * 60}")
    print("""
  Current LightGBM Config:
    num_leaves=63, learning_rate=0.05, n_estimators=800
    min_child_samples=20, subsample=0.8

  Recommended Search Space (Optuna):
  ┌──────────────────────┬──────────────┬────────────────┐
  │ Parameter            │ Range        │ Priority       │
  ├──────────────────────┼──────────────┼────────────────┤
  │ num_leaves           │ 31 – 127     │ ★★★ High       │
  │ learning_rate        │ 0.01 – 0.1   │ ★★★ High       │
  │ n_estimators         │ 500 – 2000   │ ★★ Medium      │
  │ min_child_samples    │ 10 – 50      │ ★★ Medium      │
  │ subsample            │ 0.6 – 1.0    │ ★ Low          │
  │ colsample_bytree     │ 0.6 – 1.0    │ ★ Low          │
  │ reg_alpha             │ 0 – 1.0     │ ★ Low          │
  │ reg_lambda            │ 0 – 1.0     │ ★ Low          │
  └──────────────────────┴──────────────┴────────────────┘

  Current XGBoost Config:
    max_depth=6, learning_rate=0.05, n_estimators=600
    subsample=0.8, colsample_bytree=0.8

  💡 Tips:
  • Use walk_forward_cv() for validation — NOT random split
  • Optimise for logloss (calibration) not AUC
  • With 148 features, consider colsample_bytree=0.6-0.8
  • Use early stopping with patience=50
""")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Feature Importance & SHAP Analysis")
    parser.add_argument("--shap", action="store_true", help="Run full SHAP analysis")
    parser.add_argument("--race", type=int, help="Explain predictions for a specific race ID")
    parser.add_argument("--top", type=int, default=30, help="Number of top features to show")
    parser.add_argument("--version", type=str, default="latest", help="Model version")
    parser.add_argument("--tune", action="store_true", help="Show hyperparameter tuning recommendations")
    args = parser.parse_args()

    if args.tune:
        suggest_hyperparameters()
        return

    if args.race:
        explain_race(race_id=args.race, version=args.version, top_n=args.top)
        return

    # Default: feature importance from saved model
    log.info("Loading feature importance from saved model...")
    imp_df = get_feature_importance(version=args.version, top_n=args.top)
    print_importance_table(imp_df)
    save_importance_json(imp_df)

    if args.shap:
        run_shap_analysis(version=args.version, top_n=args.top)


if __name__ == "__main__":
    main()
