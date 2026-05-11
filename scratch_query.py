import sys
import os
import pandas as pd
from sqlalchemy import create_engine, text

# Get DB URL from env
from dotenv import load_dotenv
load_dotenv()
db_url = os.environ.get("DATABASE_URL")
engine = create_engine(db_url)

query = """
SELECT r.post_time, s.captured_at, r.id as race_id, s.odds_value
FROM horsebet.odds_snapshots s
JOIN horsebet.races r ON s.race_id = r.id
ORDER BY s.captured_at DESC
LIMIT 50;
"""
df = pd.read_sql(query, engine)
print(df)
print("\n--- Summary ---")
print("Total snaps after post_time:")
bad_snaps = df[df["captured_at"] > pd.to_datetime(df["post_time"], utc=True)]
print(len(bad_snaps))
