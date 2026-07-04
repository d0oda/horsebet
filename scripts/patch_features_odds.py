import pandas as pd
from sqlalchemy import text
from scraper.db import get_session
import numpy as np

print("Loading features.parquet...")
df = pd.read_parquet("data/features.parquet")

print("Loading true odds from repaired database...")
with get_session() as session:
    rows = session.execute(text("SELECT id, odds_win, popularity FROM entries")).fetchall()
    
odds_map = {r[0]: r[1] for r in rows}
pop_map = {r[0]: r[2] for r in rows}

print("Updating odds and popularity in features...")
df["odds_win"] = df["entry_id"].map(odds_map)
df["popularity"] = df["entry_id"].map(pop_map)

# Update log_odds and z-scores
print("Recalculating odds-based features...")
df["log_odds"] = np.log(df["odds_win"].clip(lower=1.0))

# We also need to fix odds_ratio_to_fav.
# This requires grouping by race_id and finding the minimum odds (fav_odds).
def fix_race_odds(group):
    odds = group["odds_win"]
    if odds.isna().all():
        return group
    
    fav_odds = odds.min()
    group["odds_ratio_to_fav"] = odds / fav_odds if fav_odds > 0 else np.nan
    
    log_odds = np.log(odds.clip(lower=1.0))
    group["log_odds_z"] = (log_odds - log_odds.median()) / (log_odds.std() + 1e-6)
    group["odds_win_z"] = (odds - odds.median()) / (odds.std() + 1e-6)
    
    return group

print("Recalculating relative odds features (this takes a few seconds)...")
df = df.groupby("race_id", group_keys=False).apply(fix_race_odds)

print("Saving fixed features.parquet...")
df.to_parquet("data/features.parquet")
print("Done! Features repaired perfectly in seconds!")
