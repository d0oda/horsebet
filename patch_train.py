import sys
import re

with open("models/train.py", "r") as f:
    content = f.read()

# 1. Add evaluation metrics for regression
eval_imports = "from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score"
eval_imports_new = "from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, mean_squared_error, mean_absolute_error"
content = content.replace(eval_imports, eval_imports_new)

# 2. Add regression training functions
regression_funcs = """
def train_lightgbm_regression(X_train, y_train, X_val, y_val, feature_cols) -> tuple:
    \"\"\"Train a LightGBM regression model for Beaten Lengths.\"\"\"
    lgb = _get_lgb()

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": 42,
    }

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mae = mean_absolute_error(y_val, y_pred)
    log.info(f"LightGBM Reg — RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return model, y_pred

def train_xgboost_regression(X_train, y_train, X_val, y_val, feature_cols) -> tuple:
    \"\"\"Train an XGBoost regression model for Beaten Lengths.\"\"\"
    xgb = _get_xgb()

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    y_pred = model.predict(dval)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mae = mean_absolute_error(y_val, y_pred)
    log.info(f"XGBoost Reg  — RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    return model, y_pred

# ---------------------------------------------------------------------------
# Learning-to-Rank Training
"""

content = content.replace("# ---------------------------------------------------------------------------\n# Learning-to-Rank Training", regression_funcs)

# 3. Add training calls in main()
training_section = """    # Train models
    log.info("\\n--- Training LightGBM ---")
    lgb_model, lgb_preds = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols)

    log.info("\\n--- Training XGBoost ---")
    xgb_model, xgb_preds = train_xgboost(X_train, y_train, X_val, y_val, feature_cols)

    # Ensemble
    log.info("\\n--- Ensemble ---")
    ensemble_preds = ensemble_predict(lgb_preds, xgb_preds)"""

training_section_new = """    # Train binary classifiers
    log.info("\\n--- Training LightGBM ---")
    lgb_model, lgb_preds = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols)

    log.info("\\n--- Training XGBoost ---")
    xgb_model, xgb_preds = train_xgboost(X_train, y_train, X_val, y_val, feature_cols)
    
    # Train regression models on Beaten Lengths (target_margin)
    log.info("\\n--- Training Regression Models (Beaten Lengths) ---")
    X_train_reg, y_train_reg, X_val_reg, y_val_reg, _, race_ids_val_reg = prepare_data(
        df, target="target_margin", val_date=args.val_date, test_date=args.test_date,
        exclude_features=exclude,
    )
    if len(X_train_reg) > 50:
        lgb_reg_model, lgb_reg_preds = train_lightgbm_regression(X_train_reg, y_train_reg, X_val_reg, y_val_reg, feature_cols)
        xgb_reg_model, xgb_reg_preds = train_xgboost_regression(X_train_reg, y_train_reg, X_val_reg, y_val_reg, feature_cols)
        ensemble_reg_preds = ensemble_predict(lgb_reg_preds, xgb_reg_preds)
        
        # Convert regression margins to probabilities (lower margin = higher prob)
        reg_probs = scores_to_probs(-ensemble_reg_preds, race_ids_val_reg)
    else:
        log.warning("Not enough data to train regression models.")
        reg_probs = None

    # Ensemble (Binary Classification + Regression)
    log.info("\\n--- Ensemble ---")
    ensemble_preds = ensemble_predict(lgb_preds, xgb_preds)
    
    # If reg_probs is available and aligns perfectly (no rows dropped differently), blend them
    if reg_probs is not None and len(reg_probs) == len(ensemble_preds):
        log.info("Blending Binary classifiers (80%) and Regression probabilities (20%)")
        ensemble_preds = 0.8 * ensemble_preds + 0.2 * reg_probs
"""

content = content.replace(training_section, training_section_new)

with open("models/train.py", "w") as f:
    f.write(content)

print("Patch applied to models/train.py")
