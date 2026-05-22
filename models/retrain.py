"""
UmaEdge — Weekly Retrain Pipeline (Sprint 4.1).

Automated weekly pipeline:
  1. Scrape latest weekend races
  2. Rebuild features
  3. Retrain models with walk-forward CV
  4. Apply isotonic calibration
  5. Save model with timestamped version
  6. Log metrics to drift tracker

Usage:
    python -m models.retrain
    python -m models.retrain --year 2025 --max-races 20
"""

import argparse
import logging
import os
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retrain")


def run_retrain(
    scrape_year: int = None,
    scrape_max_races: int = 20,
    calibration_method: str = "isotonic",
    walk_forward: bool = True,
    skip_scrape: bool = False,
    use_cache: bool = False,
):
    """
    Full weekly retrain pipeline.

    Args:
        scrape_year: Year to scrape new races from (default: current year)
        scrape_max_races: Max new races to scrape
        calibration_method: Calibration to apply ('isotonic', 'platt', 'none')
        walk_forward: Use walk-forward CV
        skip_scrape: Skip the scraping step (use existing data)
        use_cache: Load features from cache if exists
    """
    from models.features import FeatureBuilder
    from models.train import (
        prepare_data, train_lightgbm, train_xgboost, ensemble_predict,
        evaluate_ensemble, save_model, calibrate_predictions,
        walk_forward_cv,
    )
    from models.drift import log_metrics

    version = datetime.now().strftime("retrain_%Y%m%d_%H%M")

    if scrape_year is None:
        scrape_year = datetime.now().year

    # Step 1: Scrape latest races
    if not skip_scrape:
        log.info(f"Step 1: Scraping latest {scrape_max_races} races from {scrape_year}...")
        try:
            from scraper.batch_scrape import batch_scrape
            stats = batch_scrape(scrape_year, max_races=scrape_max_races)
            log.info(f"  Scraped {stats['races_scraped']} new races")
        except Exception as e:
            log.warning(f"  Scraping failed: {e} (continuing with existing data)")
    else:
        log.info("Step 1: Skipping scrape (--skip-scrape)")

    # Step 2: Build features
    log.info("Step 2: Building feature vectors...")
    
    cache_path = "data/features.parquet"
    if use_cache and os.path.exists(cache_path):
        log.info(f"Loading cached features from {cache_path}...")
        df = pd.read_parquet(cache_path)
    else:
        fb = FeatureBuilder()
        df = fb.build_features_all()
        # Save cache if needed
        if not df.empty:
            log.info(f"Saving features to cache {cache_path}...")
            os.makedirs("data", exist_ok=True)
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
            df.to_parquet(cache_path, index=False)

    if df.empty:
        log.error("No data available for training")
        return None

    feature_cols = FeatureBuilder.get_feature_columns(df)
    log.info(f"  {len(df)} entries, {len(feature_cols)} features")

    # Step 3: Walk-forward CV or standard train
    if walk_forward and "date" in df.columns:
        log.info("Step 3: Running walk-forward CV...")
        cv_results = walk_forward_cv(df, feature_cols, n_folds=4)
        if cv_results:
            avg_auc = sum(f.get("auc", 0) for f in cv_results) / len(cv_results)
            avg_logloss = sum(f.get("logloss", 0) for f in cv_results) / len(cv_results)
            log.info(f"  Avg AUC: {avg_auc:.4f}")
            log.info(f"  Avg Log-loss: {avg_logloss:.4f}")

    # Step 4: Full retrain on all data
    log.info("Step 4: Training final model...")
    X_train, y_train, X_val, y_val, feature_cols, race_ids_val = prepare_data(df)

    lgb_model, lgb_preds = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols)
    xgb_model, xgb_preds = train_xgboost(X_train, y_train, X_val, y_val, feature_cols)

    ensemble_preds = ensemble_predict(lgb_preds, xgb_preds)
    metrics = evaluate_ensemble(y_val, ensemble_preds, label=version)

    # Step 5: Calibrate
    log.info(f"Step 5: Applying {calibration_method} calibration...")
    calibrator = None
    if calibration_method != "none":
        import xgboost as xgb
        # Generate raw predictions on training set for calibrator to fit
        lgb_preds_train = lgb_model.predict(X_train)
        xgb_preds_train = xgb_model.predict(xgb.DMatrix(X_train, feature_names=feature_cols))
        ensemble_preds_train = ensemble_predict(lgb_preds_train, xgb_preds_train)

        ensemble_preds, calibrator = calibrate_predictions(
            y_train=y_train,
            raw_preds_train=ensemble_preds_train,
            y_val=y_val,
            raw_preds_val=ensemble_preds,
            method=calibration_method
        )

    # Step 6: Save model
    log.info(f"Step 6: Saving model version '{version}'...")
    try:
        save_model(
            lgb_model, xgb_model, feature_cols, metrics,
            version=version, calibrator=calibrator,
        )
    except Exception as e:
        log.warning(f"  Save failed: {e}")

    # Step 7: Log drift metrics
    log.info("Step 7: Logging drift metrics...")
    log_metrics(
        version=version,
        auc=metrics.get("auc", 0),
        log_loss=metrics.get("log_loss", 0),
        brier=metrics.get("brier", 0),
        n_samples=len(y_val),
    )

    log.info(f"\n✅ Retrain complete: version='{version}'")
    log.info(f"   AUC={metrics.get('auc', 0):.4f}, "
             f"Log-loss={metrics.get('log_loss', 0):.4f}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Weekly Retrain Pipeline")
    parser.add_argument("--year", type=int, help="Year to scrape (default: current)")
    parser.add_argument("--max-races", type=int, default=20, help="Max races to scrape")
    parser.add_argument("--calibration", default="isotonic",
                        choices=["none", "platt", "isotonic"])
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip scraping, use existing data")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip walk-forward CV")
    parser.add_argument("--use-cache", action="store_true",
                        help="Load features from data/features.parquet if exists")
    args = parser.parse_args()

    run_retrain(
        scrape_year=args.year,
        scrape_max_races=args.max_races,
        calibration_method=args.calibration,
        walk_forward=not args.no_cv,
        skip_scrape=args.skip_scrape,
        use_cache=args.use_cache,
    )


if __name__ == "__main__":
    main()
