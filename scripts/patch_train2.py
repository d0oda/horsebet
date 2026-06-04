import sys
import re

with open("models/train.py", "r") as f:
    content = f.read()

# 1. Update prepare_data feature cols and categorical conversion
prepare_data_old = """    # Get feature columns (everything except IDs and targets)
    exclude = {"race_id", "entry_id", "target_win", "target_place", "finish_pos", "date", "horse_name"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, float, int]]"""

prepare_data_new = """    # Categorical columns
    cat_cols = ["sire_id", "broodmare_sire_id", "going_code", "surface_code", "draw"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Get feature columns (everything except IDs and targets)
    exclude = {"race_id", "entry_id", "target_win", "target_place", "target_margin", "finish_pos", "date", "horse_name"}
    feature_cols = [c for c in df.columns if c not in exclude]"""
content = content.replace(prepare_data_old, prepare_data_new)

# 2. Update prepare_data to return DataFrames instead of .values
split_old = """    X_train = train_df[feature_cols].values
    y_train = train_df[target].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target].values"""

split_new = """    X_train = train_df[feature_cols]
    y_train = train_df[target].values
    X_val = val_df[feature_cols]
    y_val = val_df[target].values"""
content = content.replace(split_old, split_new)

# 3. Update train_xgboost
xgb_old = """    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss","""
xgb_new = """    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols, enable_categorical=True)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist","""
content = content.replace(xgb_old, xgb_new)

# 4. Update train_xgboost_regression
xgb_reg_old = """    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse","""
xgb_reg_new = """    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols, enable_categorical=True)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist","""
content = content.replace(xgb_reg_old, xgb_reg_new)

# 5. Update predict_race
pred_old = """    # Align features (pass NaNs directly, models handle natively)
    X = race_features[feature_cols].values

    lgb_probs = lgb_model.predict(X)
    xgb_probs = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_cols))"""
pred_new = """    # Align features (pass NaNs directly, models handle natively)
    for col in ["sire_id", "broodmare_sire_id", "going_code", "surface_code", "draw"]:
        if col in race_features.columns:
            race_features[col] = race_features[col].astype("category")
    X = race_features[feature_cols]

    lgb_probs = lgb_model.predict(X)
    xgb_probs = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_cols, enable_categorical=True))"""
content = content.replace(pred_old, pred_new)

# 6. Update train calibration predict calls
# Where calibrator is used, it uses X_train directly which is now a DataFrame.
cal_old = """        xgb_train_preds = xgb_model.predict(
            xgb_lib.DMatrix(X_train, feature_names=feature_cols)
        )"""
cal_new = """        xgb_train_preds = xgb_model.predict(
            xgb_lib.DMatrix(X_train, feature_names=feature_cols, enable_categorical=True)
        )"""
content = content.replace(cal_old, cal_new)

with open("models/train.py", "w") as f:
    f.write(content)

print("Applied native categorical patch to models/train.py")
