import pandas as pd
import numpy as np
import json
import xgboost as xgb_lib
from models.train import load_model, ensemble_predict
from models.ensemble import HybridEnsemble
import os

print("Loading cached features from data/features.parquet...")
df = pd.read_parquet("data/features.parquet")
df = df.dropna(subset=["target_win", "date"]).copy()
df = df.sort_values("date")

# Same train/val logic as train.py
unique_races = df["race_id"].unique()
split_idx = int(len(unique_races) * 0.8)
val_races = set(unique_races[split_idx:])

val_df = df[df["race_id"].isin(val_races)].copy()
print(f"Validation set: {len(val_df)} entries ({len(val_races)} races) from {val_df['date'].min()} to {val_df['date'].max()}")

# 1. Standard Model Predictions
print("Generating predictions for Standard Model (retrain_20260523_2115)...")
std_version = "retrain_20260523_2115"
lgb_model, xgb_model, meta = load_model(version=std_version)
feature_cols = meta["feature_cols"]
calibrator = meta.get("calibrator")

std_df = val_df[["race_id", "entry_id", "date", "horse_name", "finish_pos", "odds_win"]].copy()

# Add missing cols as nan
for col in feature_cols:
    if col not in val_df.columns:
        val_df[col] = np.nan

X = val_df[feature_cols].values
lgb_preds = lgb_model.predict(X)
xgb_preds = xgb_model.predict(xgb_lib.DMatrix(X, feature_names=feature_cols))
combined_probs = ensemble_predict(lgb_preds, xgb_preds)

if calibrator is not None:
    combined_probs = calibrator.predict(combined_probs)

std_df["win_prob"] = combined_probs
std_df.to_parquet("data/test_preds_standard.parquet")
print("Saved data/test_preds_standard.parquet")


