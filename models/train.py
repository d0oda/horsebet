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
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, mean_squared_error, mean_absolute_error
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
    val_date: str = None,
    test_date: str = None,
    exclude_features: list = None
) -> tuple:
    """
    Split feature dataframe into train/val sets.
    Uses time-based split: train on everything before val_date, validate on val_date to test_date.
    The test_date onwards is completely quarantined from this function to prevent early-stopping leakage.

    Args:
        df: Feature dataframe.
        target: Target column name.
        val_date: Validation cutoff date (YYYY-MM-DD).
        test_date: Test cutoff date (YYYY-MM-DD). Data >= test_date is excluded.
        exclude_features: Optional list of feature names to drop
            (e.g. ODDS_FEATURES for odds-free model).

    Returns:
        (X_train, y_train, X_val, y_val, feature_cols, race_ids_val)
    """
    # Drop rows with no target
    df = df.dropna(subset=[target]).copy()

    # Categorical columns
    cat_cols = ["sire_id", "broodmare_sire_id", "going_code", "surface_code", "draw"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).astype("category")

    # Get feature columns (everything except IDs and targets)
    exclude = {"race_id", "entry_id", "target_win", "target_place", "target_margin", "finish_pos", "date", "horse_name"}
    feature_cols = [c for c in df.columns if c not in exclude]

    # Exclude specified features (e.g. odds-derived features for odds-free model)
    if exclude_features:
        before = len(feature_cols)
        feature_cols = [c for c in feature_cols if c not in set(exclude_features)]
        log.info(f"Excluded {before - len(feature_cols)} features: {exclude_features}")

    # Remove any all-NaN columns
    valid_cols = [c for c in feature_cols if not df[c].isna().all()]
    feature_cols = valid_cols

    log.info(f"Using {len(feature_cols)} features")

    # --- Split first ---
    if val_date is None:
        # Default: last 20% of races by date
        if "date" in df.columns:
            unique_races = df.sort_values("date")["race_id"].unique()
        else:
            unique_races = df["race_id"].unique()
        split_idx = int(len(unique_races) * 0.8)
        train_races = set(unique_races[:split_idx])
        val_races = set(unique_races[split_idx:])
    else:
        # Time-based split
        if "date" in df.columns:
            train_mask = df["date"] < val_date
            if test_date:
                val_mask = (df["date"] >= val_date) & (df["date"] < test_date)
                log.info(f"Quarantining data >= {test_date} for pure out-of-sample testing.")
            else:
                val_mask = df["date"] >= val_date
            train_races = set(df[train_mask]["race_id"].unique())
            val_races = set(df[val_mask]["race_id"].unique())
        else:
            unique_races = df["race_id"].unique()
            split_idx = int(len(unique_races) * 0.8)
            train_races = set(unique_races[:split_idx])
            val_races = set(unique_races[split_idx:])

    train_df = df[df["race_id"].isin(train_races)].copy()
    val_df = df[df["race_id"].isin(val_races)].copy()

    # --- Removed manual NaN imputation ---
    # LightGBM and XGBoost natively handle NaN values perfectly by finding the optimal
    # split direction. Imputing with medians destroys this capability and risks train/serve skew.

    X_train = train_df[feature_cols]
    y_train = train_df[target].values
    X_val = val_df[feature_cols]
    y_val = val_df[target].values

    log.info(f"Train: {len(X_train)} entries ({len(train_races)} races)")
    log.info(f"Val:   {len(X_val)} entries ({len(val_races)} races)")
    log.info(f"Win rate — train: {y_train.mean():.3f}, val: {y_val.mean():.3f}")

    return X_train, y_train, X_val, y_val, feature_cols, val_df["race_id"].values, train_df["race_id"].values


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

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols, enable_categorical=True)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
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



def train_lightgbm_regression(X_train, y_train, X_val, y_val, feature_cols) -> tuple:
    """Train a LightGBM regression model for Beaten Lengths."""
    lgb = _get_lgb()

    params = {
        "objective": "regression",
        "metric": "rmse",
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

    y_pred = model.predict(X_val)
    import numpy as np
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    log.info(f"LightGBM Reg — RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return model, y_pred

def train_xgboost_regression(X_train, y_train, X_val, y_val, feature_cols) -> tuple:
    """Train an XGBoost regression model for Beaten Lengths."""
    xgb = _get_xgb()

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols, enable_categorical=True)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
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
    import numpy as np
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    log.info(f"XGBoost Reg  — RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return model, y_pred

# ---------------------------------------------------------------------------
# Learning-to-Rank Training

# ---------------------------------------------------------------------------

def _make_group_array(race_ids: np.ndarray) -> np.ndarray:
    """Convert a race_id array into a group-count array for ranking.
    
    E.g., [1,1,1,2,2,3,3,3,3] -> [3,2,4]
    Data must be pre-sorted by race_id.
    """
    _, counts = np.unique(race_ids, return_counts=True)
    return counts


def _finish_to_relevance(finish_pos: np.ndarray, max_rel: int = 5) -> np.ndarray:
    """Convert finish position to relevance label for LambdaRank.
    
    1st -> 5 (highest relevance)
    2nd -> 4
    3rd -> 3
    4th -> 2
    5th -> 1
    6th+ -> 0
    
    This gives the ranker a graded signal, not just binary win/lose.
    """
    rel = np.clip(max_rel + 1 - finish_pos, 0, max_rel)
    return rel.astype(int)


def train_lightgbm_ranker(X_train, y_train, X_val, y_val, feature_cols,
                           train_groups, val_groups) -> tuple:
    """Train a LightGBM LambdaRank model that ranks horses within each race."""
    lgb = _get_lgb()

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [1, 3],  # NDCG@1 (winner) and NDCG@3 (top 3)
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
        "label_gain": [0, 1, 2, 4, 8, 16],  # Geometric progression: 1st=16, 2nd=8, 3rd=4, 4th=2, 5th=1, 6th=0
        # Previously [0,1,2,3,4,31] — the gain(2^31)≈2.1B for 1st vs gain(2^4)≈15 for 2nd caused
        # LambdaRank to ignore 2nd-5th ranking entirely (ranker collapsed to binary win classifier).
    }

    train_set = lgb.Dataset(X_train, label=y_train, group=train_groups,
                             feature_name=feature_cols)
    val_set = lgb.Dataset(X_val, label=y_val, group=val_groups,
                           feature_name=feature_cols, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    y_pred = model.predict(X_val)

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:15]
    log.info("LightGBM Ranker — Top 15 features (gain):")
    for name, gain in top_features:
        log.info(f"  {name}: {gain:.0f}")

    return model, y_pred


def train_xgboost_ranker(X_train, y_train, X_val, y_val, feature_cols,
                          train_groups, val_groups) -> tuple:
    """Train an XGBoost LambdaMART model that ranks horses within each race."""
    xgb = _get_xgb()

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols, enable_categorical=True)
    dtrain.set_group(train_groups)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols, enable_categorical=True)
    dval.set_group(val_groups)

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@1",
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

    return model, y_pred


def scores_to_probs(scores: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """Convert raw scores to per-race probabilities via softmax normalisation.

    Use for RANKER and REGRESSION outputs (raw unbounded logits/scores).
    For CLASSIFIER outputs (already in [0,1]), use cls_to_probs() instead,
    which applies simple L1 normalisation to preserve calibrated magnitudes.

    Args:
        scores: Raw scores from the ranker or regression model.
        race_ids: Corresponding race IDs (same length as scores).

    Returns:
        Array of per-race probabilities summing to 1.0 per race.
    """
    probs = np.zeros_like(scores, dtype=np.float64)

    for rid in np.unique(race_ids):
        mask = race_ids == rid
        race_scores = scores[mask]
        # Numerical stability: subtract max before exp
        shifted = race_scores - race_scores.max()
        exp_scores = np.exp(shifted)
        probs[mask] = exp_scores / exp_scores.sum()

    return probs


def cls_to_probs(probs_in: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """L1-normalise per-race binary-classifier probabilities.

    Classifier outputs are already calibrated probabilities in [0,1].
    Softmax would flatten the distribution; simple proportional rescaling
    (divide each horse's prob by the race sum) is the correct operation here.

    Args:
        probs_in: Classifier probabilities for each horse.
        race_ids: Corresponding race IDs (same length as probs_in).

    Returns:
        Array of per-race probabilities summing to 1.0 per race.
    """
    probs = probs_in.copy().astype(np.float64)
    for rid in np.unique(race_ids):
        mask = race_ids == rid
        s = probs[mask].sum()
        if s > 0:
            probs[mask] /= s
    return probs



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
        y_train: True labels for calibration fitting (should be *validation* labels, NOT
                 training labels — training predictions are overfit and cause leakage).
        raw_preds_train: Ensemble predictions for calibration fitting (validation set).
        y_val: True labels for post-calibration metric logging.
        raw_preds_val: Ensemble predictions to transform (validation set).
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
    
    # Categorical columns
    cat_cols = ["sire_id", "broodmare_sire_id", "going_code", "surface_code", "draw"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).astype("category")

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

        # Removed manual NaN imputation here.
        # Tree-based models natively support and optimize missing value splits.

        X_train = train_df[feature_cols]
        y_train = train_df[target].values
        X_val = val_df[feature_cols]
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
    lgb_reg_model=None, xgb_reg_model=None,
    lgb_rank_model=None, xgb_rank_model=None,
    categories: Optional[dict] = None,
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
    
    # Save Regression Models
    if lgb_reg_model is not None:
        lgb_reg_model.save_model(str(model_dir / "lgb_reg_model.txt"))
    if xgb_reg_model is not None:
        xgb_reg_model.save_model(str(model_dir / "xgb_reg_model.json"))

    # Save Ranker Models
    if lgb_rank_model is not None:
        lgb_rank_model.save_model(str(model_dir / "lgb_rank_model.txt"))
    if xgb_rank_model is not None:
        xgb_rank_model.save_model(str(model_dir / "xgb_rank_model.json"))

    # Save calibrator
    if calibrator is not None:
        with open(model_dir / "calibrator.pkl", "wb") as f:
            pickle.dump(calibrator, f)

    # Save metadata — including category mappings for enforcement at inference
    # (A2 fix: category int codes must match between training and prediction).
    cat_cols_saved = ["sire_id", "broodmare_sire_id", "going_code", "surface_code", "draw"]
    meta = {
        "version": version,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "odds_free": odds_free,
        "calibration_method": calibration_method,
        "created_at": datetime.now().isoformat(),
        "categories": categories or {},
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
        versions = [d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("202")]
        if not versions:
            raise FileNotFoundError("No saved models found")
        versions = sorted(versions, key=lambda x: x.name)
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

    # Load regression models if they exist
    lgb_reg_model = None
    xgb_reg_model = None
    if (model_dir / "lgb_reg_model.txt").exists():
        lgb_reg_model = lgb.Booster(model_file=str(model_dir / "lgb_reg_model.txt"))
    if (model_dir / "xgb_reg_model.json").exists():
        xgb_reg_model = xgb.Booster()
        xgb_reg_model.load_model(str(model_dir / "xgb_reg_model.json"))

    # Load ranker models if they exist
    lgb_rank_model = None
    xgb_rank_model = None
    if (model_dir / "lgb_rank_model.txt").exists():
        lgb_rank_model = lgb.Booster(model_file=str(model_dir / "lgb_rank_model.txt"))
    if (model_dir / "xgb_rank_model.json").exists():
        xgb_rank_model = xgb.Booster()
        xgb_rank_model.load_model(str(model_dir / "xgb_rank_model.json"))

    meta["calibrator"] = calibrator
    meta["lgb_reg_model"] = lgb_reg_model
    meta["xgb_reg_model"] = xgb_reg_model
    meta["lgb_rank_model"] = lgb_rank_model
    meta["xgb_rank_model"] = xgb_rank_model
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

    # Align features (pass NaNs directly, models handle natively)
    cat_cols_meta = ["sire_id", "broodmare_sire_id", "going_code", "surface_code", "draw"]
    categories_map = meta.get("categories", {})
    for col in cat_cols_meta:
        if col in race_features.columns:
            val = race_features[col].fillna("Unknown").astype(str)
            saved_cats = categories_map.get(col)
            if saved_cats:
                # Enforce training category mapping so integer codes match the model
                race_features[col] = pd.Categorical(val, categories=saved_cats)
            else:
                race_features[col] = val.astype("category")

    import numpy as np
    missing_cols = set(feature_cols) - set(race_features.columns)
    for col in missing_cols:
        race_features[col] = np.nan

    X = race_features[feature_cols]

    lgb_probs = lgb_model.predict(X)
    xgb_probs = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
    ensemble_probs = ensemble_predict(lgb_probs, xgb_probs)

    # Normalise classifier probs per-race so that all blend components are on the
    # same scale (sum to 1.0 per race) before combining. Without this, the
    # 0.7/0.2/0.1 blend weights are misleading due to scale mismatch.
    race_ids = race_features["race_id"].values
    # Use L1 normalisation (not softmax) for classifier outputs: classifier probs
    # are already in [0,1] and softmax would flatten the distribution.
    prob_cls = cls_to_probs(ensemble_probs, race_ids)

    # Regression blend
    lgb_reg_model = meta.get("lgb_reg_model")
    xgb_reg_model = meta.get("xgb_reg_model")
    prob_reg = np.zeros_like(prob_cls)
    if lgb_reg_model is not None and xgb_reg_model is not None:
        lgb_reg_preds = lgb_reg_model.predict(X)
        xgb_reg_preds = xgb_reg_model.predict(xgb.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
        ensemble_reg_preds = ensemble_predict(lgb_reg_preds, xgb_reg_preds)
        # target_margin = 1/finish_pos (higher = better). Pass positive values.
        ensemble_reg_preds = np.maximum(ensemble_reg_preds, 0.0)
        prob_reg = scores_to_probs(ensemble_reg_preds, race_ids)

    # Ranker blend (was previously ignored in predict_race, now consistent with predict_final.py)
    lgb_rank_model = meta.get("lgb_rank_model")
    xgb_rank_model = meta.get("xgb_rank_model")
    prob_rnk = np.zeros_like(prob_cls)
    if lgb_rank_model is not None and xgb_rank_model is not None:
        lgb_rank_preds = lgb_rank_model.predict(X)
        xgb_rank_preds = xgb_rank_model.predict(xgb.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
        ensemble_rank_preds = ensemble_predict(lgb_rank_preds, xgb_rank_preds)
        prob_rnk = scores_to_probs(ensemble_rank_preds, race_ids)

    # Goldilocks blend matching predict_final.py weights (0.7 Cls / 0.2 Reg / 0.1 Rnk)
    ensemble_probs = 0.7 * prob_cls + 0.2 * prob_reg + 0.1 * prob_rnk

    calibrator = meta.get("calibrator")
    if calibrator is not None:
        cal_method = meta.get("calibration_method", "isotonic")
        if cal_method == "platt":
            ensemble_probs = calibrator.predict_proba(ensemble_probs.reshape(-1, 1))[:, 1]
        elif cal_method == "isotonic":
            ensemble_probs = calibrator.predict(ensemble_probs)

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
    parser.add_argument("--test-date", type=str, help="Test quarantine split date (YYYY-MM-DD)")
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

    cache_path = "data/features.parquet"
    if os.path.exists(cache_path):
        log.info(f"Loading cached features from {cache_path}")
        import pandas as pd
        df = pd.read_parquet(cache_path)
    else:
        log.info("No cache found. Building features from scratch...")
        fb = FeatureBuilder()
        df = fb.build_features_all()
        if not df.empty:
            log.info("Saving features to cache data/features.parquet...")
            os.makedirs("data", exist_ok=True)
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
            df.to_parquet("data/features.parquet", index=False)

    if df.empty:
        log.error("No training data — run the scraper first")
        return

    exclude = ODDS_FEATURES if odds_free else None

    # --- Walk-Forward CV (optional) ---
    if args.walk_forward:
        log.info("\n--- Walk-Forward Cross-Validation ---")
        # Prepare feature cols for CV (need to compute once)
        _, _, _, _, wf_feature_cols, _, _ = prepare_data(
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
    X_train, y_train, X_val, y_val, feature_cols, race_ids_val, race_ids_train = prepare_data(
        df, target="target_win", val_date=args.val_date, test_date=args.test_date,
        exclude_features=exclude,
    )

    # Capture category mappings from training df BEFORE any test-only rows are dropped.
    # These are used at inference to assign the same integer codes the model was trained on.
    cat_cols_for_save = ["sire_id", "broodmare_sire_id", "going_code", "surface_code", "draw"]
    training_categories = {}
    for col in cat_cols_for_save:
        if col in X_train.columns and hasattr(X_train[col], "cat"):
            training_categories[col] = X_train[col].cat.categories.tolist()

    if len(X_train) < 50:
        log.error(f"Not enough training data ({len(X_train)} entries). Need at least 50.")
        return

    # Train binary classifiers
    log.info("\n--- Training LightGBM ---")
    lgb_model, lgb_preds = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols)

    log.info("\n--- Training XGBoost ---")
    xgb_model, xgb_preds = train_xgboost(X_train, y_train, X_val, y_val, feature_cols)
    
    # Train regression models on Beaten Lengths (target_margin)
    log.info("\n--- Training Regression Models (Beaten Lengths) ---")
    X_train_reg, y_train_reg, X_val_reg, y_val_reg, _, race_ids_val_reg, _ = prepare_data(
        df, target="target_margin", val_date=args.val_date, test_date=args.test_date,
        exclude_features=exclude,
    )
    if len(X_train_reg) > 50:
        lgb_reg_model, lgb_reg_preds = train_lightgbm_regression(X_train_reg, y_train_reg, X_val_reg, y_val_reg, feature_cols)
        xgb_reg_model, xgb_reg_preds = train_xgboost_regression(X_train_reg, y_train_reg, X_val_reg, y_val_reg, feature_cols)
        ensemble_reg_preds = ensemble_predict(lgb_reg_preds, xgb_reg_preds)
        
        # target_margin = 1/finish_pos: higher prediction → better horse.
        # Clamp to 0 to prevent edge-case negative predictions from reg:squarederror
        # from inverting the ranking when passed to scores_to_probs.
        ensemble_reg_preds = np.maximum(ensemble_reg_preds, 0.0)
        reg_probs = scores_to_probs(ensemble_reg_preds, race_ids_val_reg)
    else:
        log.warning("Not enough data to train regression models.")
        reg_probs = None

    # Train Ranker models on finish_pos
    log.info("\n--- Training Ranker Models (NDCG) ---")
    X_train_rnk, y_train_rnk, X_val_rnk, y_val_rnk, _, race_ids_val_rnk, race_ids_train_rnk = prepare_data(
        df, target="finish_pos", val_date=args.val_date, test_date=args.test_date,
        exclude_features=exclude,
    )
    if len(X_train_rnk) > 50:
        y_train_rel = _finish_to_relevance(y_train_rnk)
        y_val_rel = _finish_to_relevance(y_val_rnk)
        
        # Sort by race_id for ranking groups
        sort_train = np.argsort(race_ids_train_rnk)
        X_train_rnk = X_train_rnk.iloc[sort_train]
        y_train_rel = y_train_rel[sort_train]
        race_ids_train_rnk = race_ids_train_rnk[sort_train]
        train_groups = _make_group_array(race_ids_train_rnk)
        
        sort_val = np.argsort(race_ids_val_rnk)
        X_val_rnk = X_val_rnk.iloc[sort_val]
        y_val_rel = y_val_rel[sort_val]
        race_ids_val_rnk = race_ids_val_rnk[sort_val]
        val_groups = _make_group_array(race_ids_val_rnk)
        
        lgb_rank_model, lgb_rank_preds = train_lightgbm_ranker(
            X_train_rnk, y_train_rel, X_val_rnk, y_val_rel, feature_cols, train_groups, val_groups
        )
        xgb_rank_model, xgb_rank_preds = train_xgboost_ranker(
            X_train_rnk, y_train_rel, X_val_rnk, y_val_rel, feature_cols, train_groups, val_groups
        )
    else:
        log.warning("Not enough data to train ranker models.")
        lgb_rank_model = None
        xgb_rank_model = None

    # Ensemble: align to the same 0.7/0.2/0.1 blend used in predict_final.py
    # so the calibrator is fitted on the *exact same distribution* it will correct at inference.
    log.info("\n--- Ensemble ---")
    # Per-race normalise the binary classifier probabilities before blending so that
    # the 0.7/0.2/0.1 weights are meaningful (both components become proper per-race
    # probability distributions before being combined).
    ensemble_preds = ensemble_predict(lgb_preds, xgb_preds)
    # Use L1 normalisation for classifier probs (not softmax) so the
    # 0.7/0.2/0.1 blend weights apply to correctly-scaled probability vectors.
    ensemble_preds = cls_to_probs(ensemble_preds, race_ids_val)  # normalise per-race

    # Regression component
    # A3 fix: if lengths differ (caused by NaN targets being dropped in prepare_data
    # with different target columns), fall back to zeros rather than corrupt the blend.
    # Root fix: the caller should pre-drop rows with any missing target before splitting.
    if reg_probs is not None and len(reg_probs) == len(ensemble_preds):
        blend_reg = reg_probs
    else:
        if reg_probs is not None:
            log.warning(
                f"Regression val length mismatch ({len(reg_probs)} vs {len(ensemble_preds)}). "
                "Pre-drop rows with missing targets before calling train_models to fix this."
            )
        blend_reg = np.zeros_like(ensemble_preds)

    # Ranker component (compute per-race softmax of ranker val scores)
    rnk_probs = np.zeros_like(ensemble_preds)
    try:
        import xgboost as xgb_lib  # noqa: F811
        lgb_rnk = locals().get("lgb_rank_model")
        xgb_rnk = locals().get("xgb_rank_model")
        if lgb_rnk is not None and xgb_rnk is not None:
            lgb_rank_preds_val = lgb_rnk.predict(X_val_rnk)
            dval_rnk = xgb_lib.DMatrix(X_val_rnk, enable_categorical=True)
            xgb_rank_preds_val = xgb_rnk.predict(dval_rnk)
            ens_rank_preds = ensemble_predict(lgb_rank_preds_val, xgb_rank_preds_val)
            # Re-align to race_ids_val order (ranker data was sorted)
            rnk_probs_sorted = scores_to_probs(ens_rank_preds, race_ids_val_rnk)
            if len(rnk_probs_sorted) == len(ensemble_preds):
                # Undo the ranker sort to put back into race_ids_val order
                unsort_val = np.argsort(sort_val)
                rnk_probs = rnk_probs_sorted[unsort_val]
    except Exception as e:
        log.warning(f"Could not compute ranker probs for blend: {e}")

    log.info("Blending: 70% Binary Classifier + 20% Regression + 10% Ranker")
    ensemble_preds = 0.7 * ensemble_preds + 0.2 * blend_reg + 0.1 * rnk_probs


    # Calibration
    # The calibrator is fitted on the FIRST HALF of val races and evaluated on the
    # SECOND HALF (held out from fitting). This avoids the isotonic regression
    # overfit problem where fitting and evaluating on the same data produces
    # near-perfect interpolation and a meaningless cal_logloss.
    calibrator = None
    if args.calibration != "none":
        log.info(f"\n--- Calibration ({args.calibration}) ---")
        # Split val set 50/50 by race (not by row) to avoid data leakage
        unique_val_races = np.unique(race_ids_val)
        n_cal = max(1, len(unique_val_races) // 2)
        cal_races = set(unique_val_races[:n_cal])
        hold_races = set(unique_val_races[n_cal:])

        cal_mask = np.array([r in cal_races for r in race_ids_val])
        hold_mask = np.array([r in hold_races for r in race_ids_val])

        if cal_mask.sum() > 0 and hold_mask.sum() > 0:
            ensemble_cal_preds, calibrator = calibrate_predictions(
                y_val[cal_mask], ensemble_preds[cal_mask],   # fit on first 50% of races
                y_val[hold_mask], ensemble_preds[hold_mask], # evaluate on held-out 50%
                method=args.calibration,
            )
            # Apply calibrator to full val set for downstream metric reporting
            if calibrator is not None:
                from sklearn.linear_model import LogisticRegression as LR
                if isinstance(calibrator, LR):
                    ensemble_preds = calibrator.predict_proba(ensemble_preds.reshape(-1, 1))[:, 1]
                else:
                    ensemble_preds = calibrator.predict(ensemble_preds)
        else:
            log.warning("Val set too small to split for calibration — skipping calibration")

    metrics = evaluate_ensemble(y_val, ensemble_preds)

    # Save
    version_suffix = "odds_free" if odds_free else None
    # Extract regression models from local scope if they exist
    lgb_reg_model_save = locals().get("lgb_reg_model", None)
    xgb_reg_model_save = locals().get("xgb_reg_model", None)

    version = save_model(
        lgb_model, xgb_model, feature_cols, metrics,
        version=version_suffix,
        calibrator=calibrator,
        odds_free=odds_free,
        calibration_method=args.calibration,
        lgb_reg_model=locals().get("lgb_reg_model", None),
        xgb_reg_model=locals().get("xgb_reg_model", None),
        lgb_rank_model=locals().get("lgb_rank_model", None),
        xgb_rank_model=locals().get("xgb_rank_model", None),
        categories=training_categories,
    )
    log.info(f"\n✅ Training complete. Model version: {version}")


if __name__ == "__main__":
    main()
