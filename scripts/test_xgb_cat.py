import pandas as pd
import xgboost as xgb
import numpy as np

# Test 1: Category with None (Object)
df = pd.DataFrame({
    'sire_id': ['A', 'B', None, 'A', 'C'],
    'target': [1, 0, 1, 0, 1]
})
df['sire_id'] = df['sire_id'].astype('category')

try:
    dtrain = xgb.DMatrix(df[['sire_id']], label=df['target'], enable_categorical=True)
    print("Test 1 Passed")
except Exception as e:
    print("Test 1 Failed:", e)

# Test 2: Category after casting to string
df = pd.DataFrame({
    'sire_id': ['A', 'B', None, 'A', 'C'],
    'target': [1, 0, 1, 0, 1]
})
df['sire_id'] = df['sire_id'].astype(str).astype('category')

try:
    dtrain = xgb.DMatrix(df[['sire_id']], label=df['target'], enable_categorical=True)
    print("Test 2 Passed")
except Exception as e:
    print("Test 2 Failed:", e)
