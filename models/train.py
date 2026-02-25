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
    exclude_features: Optional[list[str]] = None,
) -> tuple:
    """
    Split feature dataframe into train/val sets.
    Uses time-based split: train on everything before val_date.

    Args:
        df: Feature dataframe.
        target: Target column name.
        val_date: Validation cutoff date (YYYY-MM-DD).
        exclude_features: Optional list of feature names to drop
            (e.g. ODDS_FEATURES for odds-free model).

    Returns:
        (X_train, y_train, X_val, y_val, feature_cols, race_ids_val)
    """
    # Drop rows with no target
    df = df.dropna(subset=[target]).copy()

    # Get feature columns (everything except IDs and targets)
    exclude = {"race_id", "entry_id", "target_win", "target_place", "finish_pos", "date", "horse_name"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, float, int]]

    # Exclude specified features (e.g. odds-derived features for odds-free model)
    if exclude_features:
        before = len(feature_cols)
        feature_cols = [c for c in feature_cols if c not in set(exclude_features)]
        log.info(f"Excluded {before - len(feature_cols)} features: {exclude_features}")

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


def calibrate_predictions(
    y_train: np.ndarray,
    raw_preds_train: np.ndarray,
    y_val: np.ndarray,
    raw_preds_val: np.ndarray,
    method: str = "isotonic",
) -> tuple:
    """
    Calibrate model predictions using isotonic regression or Platt scaling.

    Args:
        y_train: True labels for calibration fitting.
        raw_preds_train: Raw ensemble predictions on the training set.
        y_val: True labels for validation.
        raw_preds_val: Raw ensemble predictions on the validation set.
        method: 'isotonic', 'platt', or 'none'.

    Returns:
        (calibrated_val_preds, calibrator_object_or_None)
    """
    if method == "none":
        return raw_preds_val, None

    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    if method == "isotonic":
        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        calibrator.fit(raw_preds_train, y_train)
        calibrated = calibrator.predict(raw_preds_val)
        log.info("Applied isotonic regression calibration")
    elif method == "platt":
        calibrator = LogisticRegression()
        calibrator.fit(raw_preds_train.reshape(-1, 1), y_train)
        calibrated = calibrator.predict_proba(raw_preds_val.reshape(-1, 1))[:, 1]
        log.info("Applied Platt (sigmoid) calibration")
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    cal_logloss = log_loss(y_val, calibrated)
    raw_logloss = log_loss(y_val, raw_preds_val)
    log.info(f"Calibration effect — LogLoss: {raw_logloss:.4f} → {cal_logloss:.4f}")

    return calibrated, calibrator


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
# Walk-Forward Cross-Validation
# ---------------------------------------------------------------------------

def walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str = "target_win",
    n_folds: int = 5,
    min_train_races: int = 50,
) -> list[dict]:
    """
    Expanding-window walk-forward cross-validation.

    Splits races chronologically into n_folds+1 slices. For fold i,
    train on slices 0..i, validate on slice i+1.

    Args:
        df: Feature dataframe with 'date' and 'race_id' columns.
        feature_cols: Feature column names.
        target: Target column.
        n_folds: Number of validation folds.
        min_train_races: Minimum races required to start training.

    Returns:
        List of per-fold metric dicts with keys:
        logloss, auc, brier, n_train, n_val, val_start, val_end
    """
    if "date" not in df.columns:
        raise ValueError("Walk-forward CV requires a 'date' column")

    df = df.dropna(subset=[target]).copy()
    df = df.sort_values("date")

    # Get unique race dates and split into chunks
    unique_dates = sorted(df["date"].unique())
    total_dates = len(unique_dates)
    chunk_size = max(1, total_dates // (n_folds + 1))

    fold_results = []

    for fold_i in range(n_folds):
        # Train on chunks 0..(fold_i)
        train_end_idx = (fold_i + 1) * chunk_size
        # Validate on chunk (fold_i + 1)
        val_start_idx = train_end_idx
        val_end_idx = min(val_start_idx + chunk_size, total_dates)

        if val_start_idx >= total_dates:
            break

        train_cutoff = unique_dates[min(train_end_idx, total_dates - 1)]
        val_dates = unique_dates[val_start_idx:val_end_idx]

        if not val_dates:
            break

        train_df = df[df["date"] < train_cutoff].copy()
        val_df = df[df["date"].isin(val_dates)].copy()

        n_train_races = train_df["race_id"].nunique()
        if n_train_races < min_train_races:
            log.info(f"Fold {fold_i + 1}: Skipping (only {n_train_races} train races)")
            continue

        # Fill NaN
        for col in feature_cols:
            median_val = train_df[col].median()
            fill_val = median_val if not np.isnan(median_val) else 0
            train_df[col] = train_df[col].fillna(fill_val)
            val_df[col] = val_df[col].fillna(fill_val)

        X_train = train_df[feature_cols].values
        y_train = train_df[target].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target].values

        if len(X_val) < 10 or len(X_train) < 50:
            continue

        # Train LightGBM + XGBoost and ensemble
        lgb_model, lgb_preds = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols)
        xgb_model, xgb_preds = train_xgboost(X_train, y_train, X_val, y_val, feature_cols)
        ensemble_preds = ensemble_predict(lgb_preds, xgb_preds)

        metrics = evaluate_ensemble(y_val, ensemble_preds, label=f"Fold {fold_i + 1}")
        metrics["n_train"] = len(X_train)
        metrics["n_val"] = len(X_val)
        metrics["n_train_races"] = n_train_races
        metrics["n_val_races"] = val_df["race_id"].nunique()
        metrics["val_start"] = val_dates[0]
        metrics["val_end"] = val_dates[-1]

        fold_results.append(metrics)
        log.info(
            f"Fold {fold_i + 1}: train={n_train_races} races, "
            f"val={metrics['n_val_races']} races ({val_dates[0]}→{val_dates[-1]}), "
            f"AUC={metrics['auc']:.4f}"
        )

    # Summary
    if fold_results:
        aucs = [f["auc"] for f in fold_results]
        losses = [f["logloss"] for f in fold_results]
        log.info(f"\nWalk-Forward CV Summary ({len(fold_results)} folds):")
        log.info(f"  AUC:     {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        log.info(f"  LogLoss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        log.info(f"  LogLoss variance: {np.var(losses):.6f}")

    return fold_results


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(
    lgb_model, xgb_model, feature_cols, metrics,
    version=None, calibrator=None, odds_free: bool = False,
    calibration_method: str = "none",
):
    """Save trained models, calibrator, and metadata."""
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_dir = MODELS_DIR / version
    model_dir.mkdir(exist_ok=True)

    # Save LightGBM
    lgb_model.save_model(str(model_dir / "lgb_model.txt"))

    # Save XGBoost
    xgb_model.save_model(str(model_dir / "xgb_model.json"))

    # Save calibrator
    if calibrator is not None:
        with open(model_dir / "calibrator.pkl", "wb") as f:
            pickle.dump(calibrator, f)

    # Save metadata
    meta = {
        "version": version,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "odds_free": odds_free,
        "calibration_method": calibration_method,
        "created_at": datetime.now().isoformat(),
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"💾 Models saved to {model_dir}")
    return version


def load_model(version: str = "latest"):
    """Load a saved model, calibrator, and metadata by version name."""
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

    # Load calibrator if present
    calibrator_path = model_dir / "calibrator.pkl"
    calibrator = None
    if calibrator_path.exists():
        with open(calibrator_path, "rb") as f:
            calibrator = pickle.load(f)
        log.info("📐 Loaded calibrator")

    with open(model_dir / "metadata.json") as f:
        meta = json.load(f)

    meta["calibrator"] = calibrator
    log.info(f"📦 Loaded model version: {meta['version']} (odds_free={meta.get('odds_free', False)})")
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
    parser.add_argument(
        "--exclude-odds", action="store_true",
        help="Train odds-free model (exclude all odds-derived features)",
    )
    parser.add_argument(
        "--calibration", type=str, default="none",
        choices=["none", "platt", "isotonic"],
        help="Calibration method (default: none)",
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Run walk-forward CV before final training",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of walk-forward CV folds (default: 5)",
    )
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
    from models.features import FeatureBuilder, ODDS_FEATURES

    log.info("=== UmaEdge Model Training ===")
    odds_free = args.exclude_odds
    if odds_free:
        log.info("🚫 ODDS-FREE MODE — excluding all odds-derived features")

    fb = FeatureBuilder()
    df = fb.build_features_all()

    if df.empty:
        log.error("No training data — run the scraper first")
        return

    exclude = ODDS_FEATURES if odds_free else None

    # --- Walk-Forward CV (optional) ---
    if args.walk_forward:
        log.info("\n--- Walk-Forward Cross-Validation ---")
        # Prepare feature cols for CV (need to compute once)
        _, _, _, _, wf_feature_cols, _ = prepare_data(
            df.copy(), target="target_win", val_date=args.val_date,
            exclude_features=exclude,
        )
        cv_results = walk_forward_cv(
            df.copy(), wf_feature_cols, target="target_win",
            n_folds=args.n_folds,
        )
        if not cv_results:
            log.warning("Walk-forward CV produced no folds (not enough data)")

    # --- Final training ---
    X_train, y_train, X_val, y_val, feature_cols, race_ids_val = prepare_data(
        df, target="target_win", val_date=args.val_date,
        exclude_features=exclude,
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

    # Calibration
    calibrator = None
    if args.calibration != "none":
        log.info(f"\n--- Calibration ({args.calibration}) ---")
        # Need training-set predictions for calibration fitting
        lgb_train_preds = lgb_model.predict(X_train)
        import xgboost as xgb_lib
        xgb_train_preds = xgb_model.predict(
            xgb_lib.DMatrix(X_train, feature_names=feature_cols)
        )
        ensemble_train_preds = ensemble_predict(lgb_train_preds, xgb_train_preds)

        ensemble_preds, calibrator = calibrate_predictions(
            y_train, ensemble_train_preds,
            y_val, ensemble_preds,
            method=args.calibration,
        )

    metrics = evaluate_ensemble(y_val, ensemble_preds)

    # Save
    version_suffix = "odds_free" if odds_free else None
    version = save_model(
        lgb_model, xgb_model, feature_cols, metrics,
        version=version_suffix,
        calibrator=calibrator,
        odds_free=odds_free,
        calibration_method=args.calibration,
    )
    log.info(f"\n✅ Training complete. Model version: {version}")


if __name__ == "__main__":
    main()
