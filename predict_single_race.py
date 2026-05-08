import sys
import pandas as pd
from sqlalchemy import text
from scraper.db import get_session
from scraper.odds_watcher import fetch_win_odds
from models.predict_final import predict_with_filters
from models.ensemble import HybridEnsemble
from models.features import FeatureBuilder
from models.train import MODELS_DIR

def run(race_netkeiba_id):
    # Get race_id
    with get_session() as session:
        race_row = session.execute(
            text("SELECT id FROM horsebet.races WHERE netkeiba_id = :nid"),
            {"nid": race_netkeiba_id}
        ).fetchone()
        
    if not race_row:
        print(f"Race {race_netkeiba_id} not in DB.")
        return
        
    db_id = race_row.id
    
    # fetch odds
    odds = fetch_win_odds(race_netkeiba_id)
    if odds:
        with get_session() as session:
            for o in odds:
                session.execute(
                    text("""
                        UPDATE horsebet.entries
                        SET odds_win = :odds
                        WHERE race_id = :race_id AND post_position = :pp
                    """),
                    {"odds": o["odds_value"], "race_id": db_id, "pp": int(o["combination"])}
                )
            session.commit()
            print(f"Updated odds for {len(odds)} entries.")
    
    # Predict using predict_with_filters
    # Actually wait, predict_with_filters needs to write to db? Let's just predict
    version = "retrain_20260507_1645"
    
    print(f"Building features for race_id={db_id}...")
    fb = FeatureBuilder()
    features_df = fb.build_features_for_races([db_id])
    
    if features_df.empty:
        print("No features built.")
        return

    print("Predicting...")
    hybrid = HybridEnsemble.load(version=version)
    all_model_cols = set(hybrid.fund_feature_cols) | set(hybrid.mkt_feature_cols)
    for col in all_model_cols:
        if col not in features_df.columns:
            features_df[col] = 0
            
    preds = hybrid.predict(features_df)
    combined_probs = preds["combined"]
    
    features_df["_unnorm_combined"] = combined_probs
    race_sums = features_df.groupby("race_id")["_unnorm_combined"].transform("sum")
    features_df["combined_win"] = features_df["_unnorm_combined"] / race_sums.replace(0, 1)
    
    entry_ids = features_df["entry_id"].tolist()
    with get_session() as session:
        entry_rows = session.execute(
            text("SELECT id, horse_id, odds_win FROM entries WHERE id = ANY(:ids)"),
            {"ids": entry_ids}
        ).fetchall()
    entry_map = {r.id: (r.horse_id, r.odds_win or 0) for r in entry_rows}
    
    results = []
    for i, (_, row) in enumerate(features_df.iterrows()):
        entry_id = int(row["entry_id"])
        _, odds = entry_map.get(entry_id, (None, 0))
        
        combined_win = row["combined_win"]
        ev = (combined_win * odds) - 1.0 if odds > 0 else 0
        
        results.append({
            "horse_name": row.get("horse_name", f"entry_{entry_id}"),
            "win_prob": round(combined_win, 4),
            "odds": odds,
            "ev": round(ev, 4),
        })
        
    df = pd.DataFrame(results).sort_values("win_prob", ascending=False)
    print(df.to_string())

if __name__ == "__main__":
    run("202605020501")
