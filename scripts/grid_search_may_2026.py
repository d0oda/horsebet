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
    # Batch feature building day by day
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

def get_model_probs(features_df, model_version):
    lgb_model, xgb_model, meta = load_model(version=model_version)
    feature_cols = meta["feature_cols"]
    calibrator = meta.get("calibrator")

    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = np.nan

    X = features_df[feature_cols].values
    lgb_preds = lgb_model.predict(X)
    xgb_preds = xgb_model.predict(xgb_lib.DMatrix(X, feature_names=feature_cols))
    combined = ensemble_predict(lgb_preds, xgb_preds)
    
    # Regression blend
    lgb_reg_model = meta.get("lgb_reg_model")
    xgb_reg_model = meta.get("xgb_reg_model")
    if lgb_reg_model is not None and xgb_reg_model is not None:
        lgb_reg_preds = lgb_reg_model.predict(X)
        xgb_reg_preds = xgb_reg_model.predict(xgb_lib.DMatrix(X, feature_names=feature_cols))
        ensemble_reg_preds = ensemble_predict(lgb_reg_preds, xgb_reg_preds)
        
        race_ids_for_probs = features_df["race_id"].values
        reg_probs = scores_to_probs(-ensemble_reg_preds, race_ids_for_probs)
        combined = 0.8 * combined + 0.2 * reg_probs
    
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
    
    # Replace these with the actual model version directories
    models = {
        "None": "20260604_223536",
        "Platt": "20260604_232834",
        "Isotonic": "20260604_233009"
    }
    
    # We will update these manually once the models finish training
    import json
    import os
    from pathlib import Path
    
    # Auto-detect models based on latest training metadata if needed
    
    ev_thresholds = [0.10, 0.20, 0.30]
    max_odds_limits = [20.0, 30.0, 50.0, 100.0]
    
    results = []
    
    for calib_name, version in models.items():
        if not version:
            continue
        print(f"Scoring model: {calib_name} ({version})")
        features_df[f"prob_{calib_name}"] = get_model_probs(features_df, version)
        
        for ev_thresh in ev_thresholds:
            for mo in max_odds_limits:
                # Calculate EV
                ev = (features_df[f"prob_{calib_name}"] * features_df["odds"]) - 1.0
                
                # Filter bets
                bets_idx = (ev >= ev_thresh) & (features_df["odds"] >= 1.5) & (features_df["odds"] <= mo)
                bets = features_df[bets_idx]
                
                n_bets = len(bets)
                if n_bets == 0:
                    continue
                    
                winners = bets[bets["finish_pos_actual"] == 1]
                hit_rate = len(winners) / n_bets
                
                total_staked = n_bets * 1000
                total_returned = winners["odds_win_actual"].sum() * 1000
                profit = total_returned - total_staked
                roi = profit / total_staked if total_staked > 0 else 0
                
                results.append({
                    "Calibration": calib_name,
                    "EV Thresh": ev_thresh,
                    "Max Odds": mo,
                    "Bets": n_bets,
                    "Hit Rate": hit_rate,
                    "ROI": roi
                })
                
    res_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS")
    print("="*80)
    print(res_df.to_string(index=False))
    
    res_df.to_csv("grid_results.csv", index=False)
    
if __name__ == "__main__":
    main()
