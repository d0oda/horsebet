import pandas as pd
import json
from pipeline import step_frontend
from scraper.db import get_session
from sqlalchemy import text

date = "2026-05-02"
version = "retrain_20260507_1645"

with get_session() as session:
    df = pd.read_sql(
        text("""
        SELECT p.race_id, p.entry_id, p.win_prob as combined_win_prob, 
               p.place_prob as pace_place_prob, r.netkeiba_id, 
               h.name_jp as horse_name, e.odds_win as odds
        FROM horsebet.predictions p
        JOIN horsebet.races r ON p.race_id = r.id
        JOIN horsebet.entries e ON p.entry_id = e.id
        JOIN horsebet.horses h ON e.horse_id = h.id
        WHERE r.date = :d AND p.model_version = :v
        """),
        session.connection(),
        params={"d": date, "v": version}
    )

if not df.empty:
    df["market_prob"] = df["odds"].apply(lambda x: 1.0 / x if x > 0 else 0)
    df["ev"] = df["combined_win_prob"] * df["odds"] - 1.0
    
    # Recalculate is_value_bet and Kelly
    df["is_value_bet"] = (df["ev"] >= 0.3) & (df["odds"] >= 2.0) & (df["odds"] <= 30.0)
    
    def calc_kelly(row):
        b = row["odds"] - 1
        p = row["combined_win_prob"]
        q = 1 - p
        kelly = max(0, (b * p - q) / b) if b > 0 else 0
        return kelly
        
    df["kelly_fraction"] = df.apply(calc_kelly, axis=1) * 0.25
    df["recommended_stake"] = (df["kelly_fraction"] * 100000).astype(int)
    
    value_bets = df[df["is_value_bet"]]
    
    output_data = {
        "model": version,
        "date": date,
        "filters": {"ev_threshold": 0.3, "min_odds": 2.0, "max_odds": 30.0},
        "summary": {
            "total_races": df["race_id"].nunique(),
            "total_entries": len(df),
            "value_bets": len(value_bets),
        },
        "predictions": df.to_dict(orient="records"),
    }
    
    pred_path = f"results/predictions_{date}.json"
    with open(pred_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Dumped {len(df)} predictions to {pred_path}")
    
    step_frontend(pred_path, date)
else:
    print("No predictions found in DB!")
