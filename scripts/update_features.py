import os
import pandas as pd
from sqlalchemy import text
import sys

from scraper.db import get_session
from models.features import FeatureBuilder

def update_cache():
    cache_path = "data/features.parquet"
    if not os.path.exists(cache_path):
        print("Cache does not exist. Run full build.")
        return

    print(f"Loading cache from {cache_path}...")
    df = pd.read_parquet(cache_path)
    
    # Get max date from cache
    max_date = df["date"].max()
    print(f"Cache contains {len(df)} rows, newest date is {max_date}")
    
    # Find races newer than max_date in the database
    with get_session() as session:
        query = "SELECT id FROM races WHERE date > :max_date ORDER BY date"
        rows = session.execute(text(query), {"max_date": max_date}).fetchall()
        new_race_ids = [r[0] for r in rows]
        
    if not new_race_ids:
        print("No new races found in DB. Cache is up to date.")
        return
        
    print(f"Found {len(new_race_ids)} new races in DB to process.")
    
    # Build features for just the new races
    fb = FeatureBuilder()
    new_df = fb.build_features_for_races(new_race_ids)
    
    if new_df.empty:
        print("Failed to build features for new races.")
        return
        
    print(f"Built {len(new_df)} new feature rows. Appending to cache...")
    
    if "date" in new_df.columns:
        new_df["date"] = new_df["date"].astype(str)
        
    # Make sure columns align
    missing_cols = set(df.columns) - set(new_df.columns)
    for col in missing_cols:
        new_df[col] = pd.NA
        
    missing_old = set(new_df.columns) - set(df.columns)
    for col in missing_old:
        df[col] = pd.NA
        
    # Append
    combined_df = pd.concat([df, new_df], ignore_index=True)
    
    print(f"Saving updated cache ({len(combined_df)} rows) to {cache_path}...")
    combined_df.to_parquet(cache_path, index=False)
    print("Done!")

if __name__ == "__main__":
    update_cache()
