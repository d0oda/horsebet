import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb_lib
from sqlalchemy import text
from scraper.db import get_session
from models.features import FeatureBuilder
from models.train import load_model, ensemble_predict, scores_to_probs

def get_may_races():
    with get_session() as session:
        dates = session.execute(
            text("SELECT DISTINCT date FROM races WHERE date >= '2026-05-01' AND date <= '2026-05-23' ORDER BY date")
        ).fetchall()
        dates = [d[0] for d in dates]

    all_race_ids = []
    for d in dates:
        with get_session() as session:
            rows = session.execute(
                text("SELECT id FROM races WHERE date = :d ORDER BY race_number"),
                {"d": d}
            ).fetchall()
            all_race_ids.extend([r[0] for r in rows])
    return all_race_ids

def build_features(race_ids):
    fb = FeatureBuilder()
    
    with get_session() as session:
        dates = session.execute(
            text("SELECT DISTINCT date FROM races WHERE id = ANY(:rids) ORDER BY date"),
            {"rids": race_ids}
        ).fetchall()
        dates = [d[0] for d in dates]

    all_dfs = []
    for d in dates:
        print(f"Building features for {d}...")
        with get_session() as session:
            rows = session.execute(
                text("SELECT id FROM races WHERE date = :d ORDER BY race_number"),
                {"d": d}
            ).fetchall()
            rids = [r[0] for r in rows if r[0] in race_ids]
            if rids:
                df = fb.build_features_for_races(rids)
                all_dfs.append(df)
                
    return pd.concat(all_dfs, ignore_index=True)

def get_raw_model_probs(features_df, model_version):
    lgb_model, xgb_model, meta = load_model(version=model_version)
    feature_cols = meta["feature_cols"]
    calibrator = meta.get("calibrator")

    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = np.nan

    X = features_df[feature_cols]
    lgb_preds = lgb_model.predict(X)
    xgb_preds = xgb_model.predict(xgb_lib.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
    prob_cls = ensemble_predict(lgb_preds, xgb_preds)
    
    race_ids = features_df["race_id"].values
    
    lgb_reg_model = meta.get("lgb_reg_model")
    xgb_reg_model = meta.get("xgb_reg_model")
    if lgb_reg_model is not None and xgb_reg_model is not None:
        lgb_reg_preds = lgb_reg_model.predict(X)
        xgb_reg_preds = xgb_reg_model.predict(xgb_lib.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
        ensemble_reg_preds = ensemble_predict(lgb_reg_preds, xgb_reg_preds)
        prob_reg = scores_to_probs(-ensemble_reg_preds, race_ids)
    else:
        prob_reg = np.zeros_like(prob_cls)
        
    lgb_rank_model = meta.get("lgb_rank_model")
    xgb_rank_model = meta.get("xgb_rank_model")
    if lgb_rank_model is not None and xgb_rank_model is not None:
        lgb_rank_preds = lgb_rank_model.predict(X)
        xgb_rank_preds = xgb_rank_model.predict(xgb_lib.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
        ensemble_rank_preds = ensemble_predict(lgb_rank_preds, xgb_rank_preds)
        prob_rnk = scores_to_probs(ensemble_rank_preds, race_ids)
    else:
        prob_rnk = np.zeros_like(prob_cls)
        
    return prob_cls, prob_reg, prob_rnk, calibrator

def apply_blend(features_df, prob_cls, prob_reg, prob_rnk, calibrator):
    combined = 0.7 * prob_cls + 0.2 * prob_reg + 0.1 * prob_rnk
    
    if calibrator is not None:
        from sklearn.linear_model import LogisticRegression
        if isinstance(calibrator, LogisticRegression):
            combined = calibrator.predict_proba(combined.reshape(-1, 1))[:, 1]
        else:
            combined = calibrator.predict(combined)

    # Normalize within race
    normed = []
    for rid, group in features_df.groupby("race_id"):
        probs = combined[group.index]
        s = sum(probs)
        normed.extend(probs / s if s > 0 else probs)
        
    return np.array(normed)

def main():
    print("Loading races...")
    race_ids = get_may_races()
    
    print(f"Extracting features for {len(race_ids)} races...")
    features_df = build_features(race_ids)
    
    print("Fetching actual results...")
    with get_session() as session:
        entries = pd.read_sql(
            text("SELECT id as entry_id, finish_pos as finish_pos_actual, odds_win as odds_win_actual FROM entries WHERE race_id = ANY(:rids)"),
            session.bind,
            params={"rids": race_ids}
        )
    
    features_df = features_df.merge(entries, on="entry_id", how="left")
    features_df["odds"] = features_df["odds_win_actual"].fillna(0)
    
    for col in ["sire_id", "broodmare_sire_id", "going_code", "surface_code", "draw"]:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(str).replace("nan", "Unknown").fillna("Unknown").astype("category")
            
    # Auto-detect latest models for each calibration method
    models = {}
    for d in sorted(Path("models/saved").iterdir(), key=lambda x: x.name):
        if d.is_dir() and d.name.startswith("202"):
            meta_path = d / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    cal_method = meta.get("calibration_method", "none")
                    models[cal_method] = d.name
    
    print(f"Found models: {models}")
    
    ev_thresholds = [0.10, 0.20, 0.30, 0.40]
    max_odds_limits = [20.0, 30.0, 50.0, 100.0]
    results = []
    
    for cal_method in ["none", "isotonic", "platt"]:
        version = models.get(cal_method)
        if not version:
            continue
            
        print(f"Testing calibrator: {cal_method} ({version})")
        prob_cls, prob_reg, prob_rnk, calibrator = get_raw_model_probs(features_df, version)
        combined_probs = apply_blend(features_df, prob_cls, prob_reg, prob_rnk, calibrator)
        features_df[f"prob"] = combined_probs
        
        for ev_thresh in ev_thresholds:
            for mo in max_odds_limits:
                ev = (features_df["prob"] * features_df["odds"]) - 1.0
                
                bets_idx = (ev >= ev_thresh) & (features_df["odds"] >= 1.5) & (features_df["odds"] <= mo)
                bets = features_df[bets_idx].copy()
                bets["ev_calc"] = ev[bets_idx]
                
                if not bets.empty:
                    # Enforce max 1 bet per race
                    best_idx = bets.groupby("race_id")["ev_calc"].idxmax()
                    bets = bets.loc[best_idx]
                
                n_bets = len(bets)
                roi = 0.0
                profit = 0.0
                if n_bets > 0:
                    winners = bets[bets["finish_pos_actual"] == 1]
                    total_staked = n_bets * 1000
                    total_returned = winners["odds_win_actual"].sum() * 1000
                    profit = total_returned - total_staked
                    roi = profit / total_staked
                    
                results.append({
                    "Calibrator": cal_method,
                    "EV Thresh": ev_thresh,
                    "Max Odds": mo,
                    "Bets": n_bets,
                    "Profit": profit,
                    "ROI": roi
                })
                
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("ROI", ascending=False)
    
    print("\n" + "="*80)
    print("TOP 15 FILTER COMBINATIONS BY ROI")
    print("="*80)
    print(res_df.head(15).to_string(index=False))
    
    res_df.to_csv("filter_results.csv", index=False)
    
if __name__ == "__main__":
    main()
