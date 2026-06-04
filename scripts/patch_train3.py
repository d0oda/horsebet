import sys
import re

with open("models/train.py", "r") as f:
    content = f.read()

# 1. Update save_model signature and body
save_old = """def save_model(
    lgb_model, xgb_model, feature_cols, metrics,
    version=None, calibrator=None, odds_free: bool = False,
    calibration_method: str = "none",
):"""
save_new = """def save_model(
    lgb_model, xgb_model, feature_cols, metrics,
    version=None, calibrator=None, odds_free: bool = False,
    calibration_method: str = "none",
    lgb_reg_model=None, xgb_reg_model=None,
):"""
content = content.replace(save_old, save_new)

save_body_old = """    # Save XGBoost
    xgb_model.save_model(str(model_dir / "xgb_model.json"))"""
save_body_new = """    # Save XGBoost
    xgb_model.save_model(str(model_dir / "xgb_model.json"))
    
    # Save Regression Models
    if lgb_reg_model is not None:
        lgb_reg_model.save_model(str(model_dir / "lgb_reg_model.txt"))
    if xgb_reg_model is not None:
        xgb_reg_model.save_model(str(model_dir / "xgb_reg_model.json"))"""
content = content.replace(save_body_old, save_body_new)

# 2. Update load_model
load_old = """    meta["calibrator"] = calibrator
    log.info(f"📦 Loaded model version: {meta['version']} (odds_free={meta.get('odds_free', False)})")
    return lgb_model, xgb_model, meta"""
load_new = """    # Load regression models if they exist
    lgb_reg_model = None
    xgb_reg_model = None
    if (model_dir / "lgb_reg_model.txt").exists():
        lgb_reg_model = lgb.Booster(model_file=str(model_dir / "lgb_reg_model.txt"))
    if (model_dir / "xgb_reg_model.json").exists():
        xgb_reg_model = xgb.Booster()
        xgb_reg_model.load_model(str(model_dir / "xgb_reg_model.json"))

    meta["calibrator"] = calibrator
    meta["lgb_reg_model"] = lgb_reg_model
    meta["xgb_reg_model"] = xgb_reg_model
    log.info(f"📦 Loaded model version: {meta['version']} (odds_free={meta.get('odds_free', False)})")
    return lgb_model, xgb_model, meta"""
content = content.replace(load_old, load_new)

# 3. Update predict_race
pred_old = """    lgb_probs = lgb_model.predict(X)
    xgb_probs = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
    ensemble_probs = ensemble_predict(lgb_probs, xgb_probs)"""
pred_new = """    lgb_probs = lgb_model.predict(X)
    xgb_probs = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
    ensemble_probs = ensemble_predict(lgb_probs, xgb_probs)
    
    # Regression blend
    lgb_reg_model = meta.get("lgb_reg_model")
    xgb_reg_model = meta.get("xgb_reg_model")
    if lgb_reg_model is not None and xgb_reg_model is not None:
        lgb_reg_preds = lgb_reg_model.predict(X)
        xgb_reg_preds = xgb_reg_model.predict(xgb.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
        ensemble_reg_preds = ensemble_predict(lgb_reg_preds, xgb_reg_preds)
        
        # Convert margin to prob proxy (negative margin so lower is better)
        race_ids = race_features["race_id"].values
        # Assuming scores_to_probs is available in models.train, if not we inline a softmax
        # But scores_to_probs is defined in models.train
        reg_probs = scores_to_probs(-ensemble_reg_preds, race_ids)
        ensemble_probs = 0.8 * ensemble_probs + 0.2 * reg_probs"""
content = content.replace(pred_old, pred_new)

# 4. Update main() save_model call
main_save_old = """    version = save_model(
        lgb_model, xgb_model, feature_cols, metrics,
        version=version_suffix,
        calibrator=calibrator,
        odds_free=odds_free,
        calibration_method=args.calibration,
    )"""
main_save_new = """    # Extract regression models from local scope if they exist
    lgb_reg_model_save = locals().get("lgb_reg_model", None)
    xgb_reg_model_save = locals().get("xgb_reg_model", None)

    version = save_model(
        lgb_model, xgb_model, feature_cols, metrics,
        version=version_suffix,
        calibrator=calibrator,
        odds_free=odds_free,
        calibration_method=args.calibration,
        lgb_reg_model=lgb_reg_model_save,
        xgb_reg_model=xgb_reg_model_save,
    )"""
content = content.replace(main_save_old, main_save_new)

with open("models/train.py", "w") as f:
    f.write(content)

print("Applied save/load patch for regression models to models/train.py")
