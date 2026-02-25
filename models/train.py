"""
UmaEdge — Model Training Pipeline.

Ensemble probability model using LightGBM + XGBoost.
Outputs calibrated P(win) and P(place) per entry.

Usage:
    # Train on all historical data
    python -m models.train

    # Train with a specific validation cutoff
    python -m models.train --val-date 2025-07-01

    # Predict for an upcoming race
    python -m models.train --predict --race-id 123
"""

import argparse
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train")

# Lazy imports (heavy libs)
_lgb = None
_xgb = None


def _get_lgb():
    global _lgb
    if _lgb is None:
        import lightgbm as lgb
        _lgb = lgb
    return _lgb


def _get_xgb():
    global _xgb
    if _xgb is None:
        import xgboost as xgb
        _xgb = xgb
    return _xgb


MODELS_DIR = Path(__file__).parent / "saved"
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def prepare_data(
    df: pd.DataFrame,
    target: str = "target_win",
    val_date: Optional[str] = None,
) -> tuple:
    """
    Split feature dataframe into train/val sets.
    Uses time-based split: train on everything before val_date.

    Returns:
        (X_train, y_train, X_val, y_val, feature_cols, race_ids_val)
    """
    # Drop rows with no target
    df = df.dropna(subset=[target]).copy()

    # Get feature columns (everything except IDs and targets)
    exclude = {"race_id", "entry_id", "target_win", "target_place", "finish_pos", "date"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, float, int]]

    # Remove any all-NaN columns
    valid_cols = [c for c in feature_cols if not df[c].isna().all()]
    feature_cols = valid_cols

    log.info(f"Using {len(feature_cols)} features")

    # Fill remaining NaN with column median
    for col in feature_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)

    # Split
    if val_date is None:
        # Default: last 20% of races by date
        unique_races = df["race_id"].unique()
        split_idx = int(len(unique_races) * 0.8)
        train_races = set(unique_races[:split_idx])
        val_races = set(unique_races[split_idx:])
    else:
        # Time-based split
        if "date" in df.columns:
            train_mask = df["date"] < val_date
            val_mask = df["date"] >= val_date
            train_races = set(df[train_mask]["race_id"].unique())
            val_races = set(df[val_mask]["race_id"].unique())
        else:
            unique_races = df["race_id"].unique()
            split_idx = int(len(unique_races) * 0.8)
            train_races = set(unique_races[:split_idx])
            val_races = set(unique_races[split_idx:])

    train_df = df[df["race_id"].isin(train_races)]
    val_df = df[df["race_id"].isin(val_races)]

    X_train = train_df[feature_cols].values
    y_train = train_df[target].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target].values

    log.info(f"Train: {len(X_train)} entries ({len(train_races)} races)")
    log.info(f"Val:   {len(X_val)} entries ({len(val_races)} races)")
    log.info(f"Win rate — train: {y_train.mean():.3f}, val: {y_val.mean():.3f}")

    return X_train, y_train, X_val, y_val, feature_cols, val_df["race_id"].values


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

def train_lightgbm(X_train, y_train, X_val, y_val, feature_cols) -> tuple:
    """Train a LightGBM binary classifier for win prediction."""
    lgb = _get_lgb()

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": 42,
    }

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # Predictions
    y_pred = model.predict(X_val)
    logloss = log_loss(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)
    brier = brier_score_loss(y_val, y_pred)

    log.info(f"LightGBM — LogLoss: {logloss:.4f}, AUC: {auc:.4f}, Brier: {brier:.4f}")

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:15]
    log.info("Top 15 features (gain):")
    for name, gain in top_features:
        log.info(f"  {name}: {gain:.0f}")

    return model, y_pred


def train_xgboost(X_train, y_train, X_val, y_val, feature_cols) -> tuple:
    """Train an XGBoost binary classifier for win prediction."""
    xgb = _get_xgb()

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    y_pred = model.predict(dval)
    logloss = log_loss(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)
    brier = brier_score_loss(y_val, y_pred)

    log.info(f"XGBoost  — LogLoss: {logloss:.4f}, AUC: {auc:.4f}, Brier: {brier:.4f}")

    return model, y_pred


# ---------------------------------------------------------------------------
# Ensemble & Calibration
# ---------------------------------------------------------------------------

def ensemble_predict(lgb_preds, xgb_preds, weights=(0.55, 0.45)):
    """Weighted average of LightGBM and XGBoost predictions."""
    return weights[0] * lgb_preds + weights[1] * xgb_preds


def evaluate_ensemble(y_true, y_pred, label="Ensemble"):
    """Evaluate probabilities with multiple metrics."""
    logloss = log_loss(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)

    log.info(f"{label} — LogLoss: {logloss:.4f}, AUC: {auc:.4f}, Brier: {brier:.4f}")

    # Calibration check: bin predictions and compare to actual win rate
    bins = np.linspace(0, 1, 11)
    binned = np.digitize(y_pred, bins)
    cal_data = []
    for b in range(1, len(bins)):
        mask = binned == b
        if mask.sum() > 0:
            predicted = y_pred[mask].mean()
            actual = y_true[mask].mean()
            cal_data.append((predicted, actual, mask.sum()))

    if cal_data:
        log.info("Calibration (predicted vs actual):")
        for pred, actual, n in cal_data:
            bar = "█" * int(actual * 50)
            log.info(f"  P={pred:.2f} → actual={actual:.3f} (n={n:>4}) {bar}")

    return {"logloss": logloss, "auc": auc, "brier": brier}


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(lgb_model, xgb_model, feature_cols, metrics, version=None):
    """Save trained models and metadata."""
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_dir = MODELS_DIR / version
    model_dir.mkdir(exist_ok=True)

    # Save LightGBM
    lgb_model.save_model(str(model_dir / "lgb_model.txt"))

    # Save XGBoost
    xgb_model.save_model(str(model_dir / "xgb_model.json"))

    # Save metadata
    meta = {
        "version": version,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "created_at": datetime.now().isoformat(),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"💾 Models saved to {model_dir}")
    return version


def load_model(version: str = "latest"):
    """Load a saved model by version name."""
    lgb = _get_lgb()
    xgb = _get_xgb()

    if version == "latest":
        versions = sorted(MODELS_DIR.iterdir())
        if not versions:
            raise FileNotFoundError("No saved models found")
        model_dir = versions[-1]
    else:
        model_dir = MODELS_DIR / version

    lgb_model = lgb.Booster(model_file=str(model_dir / "lgb_model.txt"))
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(model_dir / "xgb_model.json"))

    with open(model_dir / "metadata.json") as f:
        meta = json.load(f)

    log.info(f"📦 Loaded model version: {meta['version']}")
    return lgb_model, xgb_model, meta


def predict_race(race_features: pd.DataFrame, version: str = "latest") -> pd.DataFrame:
    """
    Generate predictions for a race using a saved model.

    Args:
        race_features: DataFrame from FeatureBuilder.build_features_for_race()
        version: Model version to use

    Returns:
        DataFrame with entry_id, win_prob, place_prob columns
    """
    xgb = _get_xgb()
    lgb_model, xgb_model, meta = load_model(version)
    feature_cols = meta["feature_cols"]

    # Align features
    X = race_features[feature_cols].fillna(0).values

    lgb_probs = lgb_model.predict(X)
    xgb_probs = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_cols))
    ensemble_probs = ensemble_predict(lgb_probs, xgb_probs)

    result = race_features[["race_id", "entry_id"]].copy()
    result["win_prob"] = ensemble_probs
    result["model_version"] = meta["version"]

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Model Training")
    parser.add_argument("--val-date", type=str, help="Validation split date (YYYY-MM-DD)")
    parser.add_argument("--predict", action="store_true", help="Predict mode (requires --race-id)")
    parser.add_argument("--race-id", type=int, help="Race ID for prediction")
    args = parser.parse_args()

    if args.predict:
        if not args.race_id:
            log.error("--race-id required for prediction mode")
            return

        from models.features import FeatureBuilder
        fb = FeatureBuilder()
        features = fb.build_features_for_race(args.race_id)
        preds = predict_race(features)
        print(preds.sort_values("win_prob", ascending=False).to_string(index=False))
        return

    # Training mode
    from models.features import FeatureBuilder

    log.info("=== UmaEdge Model Training ===")
    fb = FeatureBuilder()
    df = fb.build_features_all()

    if df.empty:
        log.error("No training data — run the scraper first")
        return

    X_train, y_train, X_val, y_val, feature_cols, race_ids_val = prepare_data(
        df, target="target_win", val_date=args.val_date
    )

    if len(X_train) < 50:
        log.error(f"Not enough training data ({len(X_train)} entries). Need at least 50.")
        return

    # Train models
    log.info("\n--- Training LightGBM ---")
    lgb_model, lgb_preds = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols)

    log.info("\n--- Training XGBoost ---")
    xgb_model, xgb_preds = train_xgboost(X_train, y_train, X_val, y_val, feature_cols)

    # Ensemble
    log.info("\n--- Ensemble ---")
    ensemble_preds = ensemble_predict(lgb_preds, xgb_preds)
    metrics = evaluate_ensemble(y_val, ensemble_preds)

    # Save
    version = save_model(lgb_model, xgb_model, feature_cols, metrics)
    log.info(f"\n✅ Training complete. Model version: {version}")


if __name__ == "__main__":
    main()
