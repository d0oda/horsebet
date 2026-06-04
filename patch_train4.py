import sys

with open("models/train.py", "r") as f:
    content = f.read()

# Fix walk_forward_cv to convert cat_cols and use DataFrames instead of .values
wf_old = """    df = df.dropna(subset=[target]).copy()
    df = df.sort_values("date")"""
wf_new = """    df = df.dropna(subset=[target]).copy()
    
    # Categorical columns
    cat_cols = ["sire_id", "broodmare_sire_id", "going_code", "surface_code", "draw"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    df = df.sort_values("date")"""
content = content.replace(wf_old, wf_new)

wf_values_old = """        # Removed manual NaN imputation here.
        # Tree-based models natively support and optimize missing value splits.

        X_train = train_df[feature_cols].values
        y_train = train_df[target].values
        X_val = val_df[feature_cols].values
        y_val = val_df[target].values"""
wf_values_new = """        # Removed manual NaN imputation here.
        # Tree-based models natively support and optimize missing value splits.

        X_train = train_df[feature_cols]
        y_train = train_df[target].values
        X_val = val_df[feature_cols]
        y_val = val_df[target].values"""
content = content.replace(wf_values_old, wf_values_new)

with open("models/train.py", "w") as f:
    f.write(content)

print("Applied fix for walk_forward_cv")
