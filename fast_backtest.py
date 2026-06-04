import argparse
import sys
from models.predict import _store_predictions, _store_value_bets
from models.features import FeatureBuilder
from scraper.db import get_session
from sqlalchemy import text
import pandas as pd
import numpy as np

def run_fast(date_str, fb, version="20260604_223536", ev_threshold=0.2):
    with get_session() as session:
        rows = session.execute(
            text("SELECT id FROM horsebet.races WHERE date = :d ORDER BY course_id, race_number"),
            {"d": date_str},
        ).fetchall()
    race_ids = [r.id for r in rows]
    if not race_ids:
        print(f"No races for {date_str}")
        return

    print(f"Building features for {date_str} ({len(race_ids)} races)...")
    features_df = fb.build_features_for_races(race_ids)

    if features_df.empty:
        return

    from models.train import MODELS_DIR, load_model, ensemble_predict
    from models.ensemble import HybridEnsemble
    import json
    meta_path = MODELS_DIR / version / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("type") == "hybrid":
        hybrid = HybridEnsemble.load(version)
        model_probs = hybrid.predict(features_df)["combined"]
    else:
        lgb_model, xgb_model, meta = load_model(version)
        feature_cols = meta["feature_cols"]
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = np.nan

        X = features_df[feature_cols].values
        lgb_probs = lgb_model.predict(X)
        import xgboost as xgb
        xgb_probs = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_cols))
        model_probs = ensemble_predict(lgb_probs, xgb_probs)
        
        # Regression blend
        lgb_reg_model = meta.get("lgb_reg_model")
        xgb_reg_model = meta.get("xgb_reg_model")
        if lgb_reg_model is not None and xgb_reg_model is not None:
            lgb_reg_preds = lgb_reg_model.predict(X)
            xgb_reg_preds = xgb_reg_model.predict(xgb.DMatrix(X, feature_names=feature_cols))
            ensemble_reg_preds = ensemble_predict(lgb_reg_preds, xgb_reg_preds)
            
            from models.train import scores_to_probs
            race_ids_for_probs = features_df["race_id"].values
            reg_probs = scores_to_probs(-ensemble_reg_preds, race_ids_for_probs)
            model_probs = 0.8 * model_probs + 0.2 * reg_probs
        
        calibrator = meta.get("calibrator")
        if calibrator is not None:
            model_probs = calibrator.predict(model_probs)

    entry_ids = features_df["entry_id"].tolist()
    with get_session() as session:
        entry_rows = session.execute(
            text("SELECT id, horse_id, odds_win FROM entries WHERE id = ANY(:ids)"),
            {"ids": entry_ids}
        ).fetchall()
    entry_map = {r.id: (r.horse_id, r.odds_win or 0) for r in entry_rows}

    combined_probs = []
    for i, (_, row) in enumerate(features_df.iterrows()):
        model_p = float(model_probs[i])
        combined_probs.append(model_p)

    features_df["_unnorm_combined"] = combined_probs
    race_sums = features_df.groupby("race_id")["_unnorm_combined"].transform("sum")
    features_df["combined_win"] = features_df["_unnorm_combined"] / race_sums.replace(0, 1)

    result_rows = []
    for i, (_, row) in enumerate(features_df.iterrows()):
        entry_id = int(row["entry_id"])
        race_id = int(row["race_id"])
        horse_id, odds = entry_map.get(entry_id, (None, 0))

        combined_win = row["combined_win"]
        market_prob = (1.0 / odds) if odds > 0 else 0
        ev = (combined_win * odds) - 1.0 if odds > 0 else 0
        
        result_rows.append({
            "race_id": race_id,
            "entry_id": entry_id,
            "horse_name": row.get("horse_name", ""),
            "model_win_prob": round(model_probs[i], 4),
            "pace_win_prob": round(model_probs[i], 4),
            "combined_win_prob": round(combined_win, 4),
            "pace_place_prob": None,
            "odds": odds,
            "market_prob": round(market_prob, 4),
            "ev": round(ev, 4),
            "is_value": ev >= ev_threshold and 2.0 <= odds <= 30.0,
            "kelly_fraction": 0,
            "recommended_stake": 1000
        })

    df = pd.DataFrame(result_rows)
    df = df.sort_values("combined_win_prob", ascending=False)

    # _store_predictions(df, version)
    # _store_value_bets(df, version, ev_threshold)
    
    # Save JSON just for analyze_april.py to work
    import json
    out = {"predictions": df.to_dict(orient="records")}
    with open(f"results/predictions_{date_str}.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    import sys
    version = sys.argv[1] if len(sys.argv) > 1 else "20260604_223536"
    ev_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.30

    dates = [
        '2026-04-04', '2026-04-05', '2026-04-11', '2026-04-12', 
        '2026-04-18', '2026-04-19', '2026-04-25', '2026-04-26'
    ]
    fb = FeatureBuilder()
    for d in dates:
        run_fast(d, fb, version=version, ev_threshold=ev_threshold)
