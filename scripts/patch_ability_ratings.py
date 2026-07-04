import pandas as pd
import numpy as np
import sqlite3
import time

t0 = time.time()
print("Loading data...")
df = pd.read_parquet("data/features.parquet")
conn = sqlite3.connect("horsebet.db")

print("Loading history...")
history = pd.read_sql("SELECT horse_id, race_date, rating_after FROM horse_rating_history ORDER BY race_date", conn)
hist_dict = {}
for _, r in history.iterrows():
    hist_dict.setdefault(r["horse_id"], []).append((r["race_date"], r["rating_after"]))

entries = pd.read_sql("SELECT id as entry_id, horse_id FROM entries", conn)
entry_to_horse = entries.set_index("entry_id")["horse_id"].to_dict()

print("Calculating PIT ratings and slopes...")
ratings = []
slopes = []

for idx, row in df.iterrows():
    eid = row["entry_id"]
    rdate = row["date"]
    hid = entry_to_horse.get(eid)
    
    horse_rating = 55.0
    slope = np.nan
    
    if hid in hist_dict:
        prior = [r for d, r in hist_dict[hid] if d < rdate]
        if prior:
            horse_rating = prior[-1]
            if len(prior) >= 3:
                last_n = prior[-5:]
                slope = np.polyfit(np.arange(len(last_n)), last_n, 1)[0]
    
    ratings.append(horse_rating)
    slopes.append(slope)

df["horse_ability_rating"] = np.round(ratings, 2)
df["rating_trend_slope"] = np.round(slopes, 4)

print("Calculating field averages...")
field_avg = df.groupby("race_id")["horse_ability_rating"].transform("mean")
df["field_avg_ability_rating"] = np.round(field_avg, 2)
df["rating_vs_field_avg"] = np.round(df["horse_ability_rating"] - field_avg, 2)
df["rating_percentile_in_field"] = np.round(df.groupby("race_id")["horse_ability_rating"].rank(pct=True), 4)

print("Calculating Z-scores...")
field_std = df.groupby("race_id")["horse_ability_rating"].transform("std")
df["rating_percentile_in_field_z"] = np.where(field_std > 0, df["rating_vs_field_avg"] / field_std, 0.0)
df["rating_percentile_in_field_z"] = np.round(df["rating_percentile_in_field_z"], 4)

print("Saving patched parquet...")
df.to_parquet("data/features.parquet")
print(f"Done in {time.time() - t0:.1f} seconds!")
