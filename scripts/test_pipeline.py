import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from models.features import FeatureBuilder
from models.train import main, predict_race
from scraper.db import get_session
from sqlalchemy import text

def run_test():
    print("Testing Feature Builder on 10 races...")
    with get_session() as session:
        rows = session.execute(text("SELECT id FROM races ORDER BY date DESC LIMIT 10")).fetchall()
        race_ids = [r[0] for r in rows]
        
    fb = FeatureBuilder()
    df = fb.build_features_for_races(race_ids)
    print(f"Built {len(df)} feature rows.")
    
    # We can't really test training effectively on just 10 races because prepare_data requires >= 50
    # But we can at least ensure predict_race runs if we have a model!
    
if __name__ == '__main__':
    run_test()
