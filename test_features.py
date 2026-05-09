from models.features import FeatureBuilder
from scraper.db import get_session
from sqlalchemy import text

# find some race ids
with get_session() as session:
    res = session.execute(text("SELECT id FROM races LIMIT 10")).fetchall()
    race_ids = [r[0] for r in res]

print(f"Testing with {len(race_ids)} races")
fb = FeatureBuilder()
df = fb.build_features_for_races(race_ids)
print(f"Features shape: {df.shape}")
