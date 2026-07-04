import pandas as pd
import numpy as np
import json
import xgboost as xgb_lib
from models.train import load_model, ensemble_predict
import os

print("Loading cached features from data/features.parquet...")
df = pd.read_parquet("data/features.parquet")
df = df.dropna(subset=["target_win", "date"]).copy()

# Categorical columns MUST be set before passing to models, as train.py does
cat_cols = ["sire_id", "broodmare_sire_id", "going_code", "surface_code", "draw"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str).astype("category")

df = df.sort_values("date")

unique_races = df["race_id"].unique()
split_idx = int(len(unique_races) * 0.8)
val_races = set(unique_races[split_idx:])

val_df = df[df["race_id"].isin(val_races)].copy()
print(f"Validation set: {len(val_df)} entries ({len(val_races)} races) from {val_df['date'].min()} to {val_df['date'].max()}")

# 1. Standard Model Predictions
print("Generating predictions for Model 20260613_162110...")
std_version = "20260613_162110"
lgb_model, xgb_model, meta = load_model(version=std_version)
feature_cols = meta["feature_cols"]
calibrator = meta.get("calibrator")

std_df = val_df[["race_id", "entry_id", "date", "horse_name", "finish_pos", "odds_win"]].copy()

# Add missing cols as nan
for col in feature_cols:
    if col not in val_df.columns:
        if col in cat_cols:
            val_df[col] = pd.Series(["Unknown"] * len(val_df), dtype="category", index=val_df.index)
        else:
            val_df[col] = np.nan

# VERY IMPORTANT: Pass the DataFrame directly to preserve 'category' dtypes, DO NOT use .values
X_df = val_df[feature_cols]

lgb_preds = lgb_model.predict(X_df)
xgb_preds = xgb_model.predict(xgb_lib.DMatrix(X_df, feature_names=feature_cols, enable_categorical=True))
combined_probs = ensemble_predict(lgb_preds, xgb_preds)

if calibrator is not None:
    # Need to reshape for calibrator if it's Platt, but models.train calibrate_predictions handled it.
    # Actually, let's just pass 1D array if it's Isotonic, which expects 1D.
    try:
        combined_probs = calibrator.predict(combined_probs)
    except ValueError:
        # LogisticRegression expects 2D
        combined_probs = calibrator.predict_proba(combined_probs.reshape(-1, 1))[:, 1]

std_df["win_prob"] = combined_probs
std_df.to_parquet("data/test_preds_standard.parquet")
print("Saved data/test_preds_standard.parquet")


