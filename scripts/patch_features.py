import pandas as pd
import numpy as np
import time

print("Loading features.parquet...")
df = pd.read_parquet("data/features.parquet")

# 1. Clear out all existing leaked odds movement features
print("Clearing leaked odds movement features...")
df["odds_morning"] = np.nan
df["odds_drift"] = np.nan
df["steam_move"] = 0
df["late_money"] = np.nan

# 2. Fetch odds snapshots from DB
from scraper.db import get_session
from sqlalchemy import text
print("Fetching odds snapshots...")
with get_session() as session:
    res = session.execute(text("SELECT race_id, captured_at, combination, odds_value FROM odds_snapshots ORDER BY captured_at")).fetchall()
    snaps_df = pd.DataFrame(res, columns=["race_id", "captured_at", "combination", "odds_value"])

print(f"Loaded {len(snaps_df)} snapshots across {snaps_df['race_id'].nunique()} races.")

# 3. Compute clean features
print("Computing clean odds movement features...")
t0 = time.time()
updates = []
groups = snaps_df.groupby("race_id")
for race_id, race_snaps in groups:
    # Filter the main feature df for this race
    race_mask = df["race_id"] == race_id
    if not race_mask.any(): continue
    
    horses_in_race = df.loc[race_mask, "entry_id"] # entry_id has nothing to do with combination
    
    # We have post_position in the parquet
    for _, horse_row in df[race_mask].iterrows():
        post = horse_row["post_position"]
        if pd.isna(post): continue
        
        # combination in odds_snapshots corresponds to post_position for win bets
        horse_snaps = race_snaps[race_snaps["combination"] == str(int(post))]
        if horse_snaps.empty: continue
            
        snaps = horse_snaps.sort_values("captured_at")
        morning_odds = float(snaps.iloc[0]["odds_value"])
        final_odds = float(snaps.iloc[-1]["odds_value"])
        
        drift = final_odds / morning_odds if morning_odds > 0 else np.nan
        steam = 1 if (not pd.isna(drift) and drift < 0.70) else 0
        
        late_money = np.nan
        if len(snaps) >= 2:
            late_money = float(snaps.iloc[-1]["odds_value"] - snaps.iloc[-2]["odds_value"])
            
        idx = horse_row.name
        df.at[idx, "odds_morning"] = morning_odds
        df.at[idx, "odds_drift"] = drift
        df.at[idx, "steam_move"] = steam
        df.at[idx, "late_money"] = late_money

print(f"Patched features in {time.time() - t0:.2f} seconds.")
print("Saving features.parquet...")
df.to_parquet("data/features.parquet")
print("Done!")
