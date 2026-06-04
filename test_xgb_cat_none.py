import pandas as pd
import xgboost as xgb
import numpy as np

# Test 3: Category with 100% None
df = pd.DataFrame({
    'sire_id': [None, None, None, None, None],
    'target': [1, 0, 1, 0, 1]
})
df['sire_id'] = df['sire_id'].astype('category')

try:
    dtrain = xgb.DMatrix(df[['sire_id']], label=df['target'], enable_categorical=True)
    print("Test 3 Passed")
except Exception as e:
    print("Test 3 Failed:", e)
