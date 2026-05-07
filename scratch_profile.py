import cProfile
import pstats
from models.features import FeatureBuilder

fb = FeatureBuilder()
# Monkey-patch to only do 2000 rows
original_load = fb._load_race_data

def fake_load(*args, **kwargs):
    df = original_load(*args, **kwargs)
    return df.head(2000)

fb._load_race_data = fake_load

cProfile.run('fb.build_features_all()', 'profile.stats')
p = pstats.Stats('profile.stats')
p.sort_stats('tottime').print_stats(20)
