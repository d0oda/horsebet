import pandas as pd
import xgboost as xgb
import numpy as np

# Test 4: Category with 100% None, filled with Unknown
df = pd.DataFrame({
    'sire_id': [None, None, None, None, None],
    'target': [1, 0, 1, 0, 1]
})
df['sire_id'] = df['sire_id'].fillna("Unknown").astype(str).astype('category')

try:
    dtrain = xgb.DMatrix(df[['sire_id']], label=df['target'], enable_categorical=True)
    print("Test 4 Passed")
except Exception as e:
    print("Test 4 Failed:", e)
