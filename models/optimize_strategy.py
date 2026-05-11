"""
Optimize Risk-Management Strategies

Runs parameter sweeps across historical out-of-sample data
to find the optimal betting configuration.
"""
import itertools
import logging
from typing import List, Dict, Any

import pandas as pd
from sqlalchemy import text
from scraper.db import get_session
from models.features import FeatureBuilder
from models.train import load_model, ensemble_predict
from models.backtest import Backtester, BacktestConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("optimize")

def get_predictions_df(model_version: str = "latest") -> pd.DataFrame:
    """Generate predictions for historical data and merge with results."""
    log.info("Loading features from data/features.parquet...")
    try:
        features_df = pd.read_parquet("data/features.parquet")
    except Exception as e:
        log.error(f"Failed to load features: {e}")
        raise ValueError("Could not load features.parquet. Run scraper/train first.")

    log.info(f"Loading model '{model_version}'...")
    lgb_model, xgb_model, meta = load_model(model_version)
    feature_cols = meta["feature_cols"]

    import xgboost as xgb
    X = features_df[feature_cols].values
    lgb_preds = lgb_model.predict(X)
    xgb_preds = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_cols))
    
    ensemble_probs = ensemble_predict(lgb_preds, xgb_preds)
    
    pred_df = features_df[["race_id", "entry_id"]].copy()
    pred_df["win_prob"] = ensemble_probs

    log.info("Fetching odds and results from DB...")
    with get_session() as session:
        data = session.execute(text("""
            SELECT
                e.id AS entry_id,
                e.odds_win,
                r.date,
                r.race_name_jp AS race_name,
                h.name_jp AS horse_name,
                res.finish_pos
            FROM entries e
            JOIN races r ON r.id = e.race_id
            JOIN horses h ON h.id = e.horse_id
            LEFT JOIN results res ON res.entry_id = e.id
        """)).fetchall()

    extra_df = pd.DataFrame(data, columns=["entry_id", "odds_win", "date", "race_name", "horse_name", "finish_pos"])
    pred_df = pred_df.merge(extra_df, on="entry_id", how="inner")
    
    return pred_df

def run_sweep():
    # 1. Prepare Data
    pred_df = get_predictions_df()
    
    # Filter to out-of-sample data (2025 onwards)
    pred_df['date'] = pd.to_datetime(pred_df['date']).dt.strftime('%Y-%m-%d')
    pred_df = pred_df[pred_df['date'] >= '2025-01-01'].copy()
    
    log.info(f"Running sweep on {len(pred_df['race_id'].unique())} races from 2025 onwards...")
    
    # 2. Define Parameter Grid
    grid = {
        'ev_threshold': [0.05, 0.10, 0.15, 0.20],
        'min_model_prob': [0.0, 0.05, 0.10, 0.15],
        'max_odds': [30.0, 50.0],
        'use_kelly': [True, False],
        'kelly_fraction': [0.10, 0.25]
    }
    
    # Generate all combinations
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Remove duplicate combinations when use_kelly=False (kelly_fraction doesn't matter)
    unique_experiments = []
    seen = set()
    for exp in experiments:
        key = (exp['ev_threshold'], exp['min_model_prob'], exp['max_odds'], exp['use_kelly'], exp['kelly_fraction'] if exp['use_kelly'] else 0.0)
        if key not in seen:
            seen.add(key)
            unique_experiments.append(exp)
            
    log.info(f"Total configurations to test: {len(unique_experiments)}")
    
    results = []
    
    for i, exp in enumerate(unique_experiments):
        config = BacktestConfig(
            ev_threshold=exp['ev_threshold'],
            min_model_prob=exp['min_model_prob'],
            max_odds=exp['max_odds'],
            use_kelly=exp['use_kelly'],
            kelly_fraction=exp['kelly_fraction'] if exp['use_kelly'] else 0.25,
            flat_stake=1000,
            initial_bankroll=100000,
            bet_type='win'
        )
        
        bt = Backtester(config)
        res = bt.run(pred_df)
        
        # Only record if we actually placed enough bets to be meaningful
        if res.total_bets >= 10:
            results.append({
                'ev_th': exp['ev_threshold'],
                'min_prob': exp['min_model_prob'],
                'max_odds': exp['max_odds'],
                'staking': f"Kelly({exp['kelly_fraction']})" if exp['use_kelly'] else "Flat",
                'bets': res.total_bets,
                'hit_rate': res.hit_rate,
                'profit': res.total_profit,
                'roi': res.roi_pct,
                'sharpe': res.sharpe,
                'max_dd_pct': res.max_drawdown_pct
            })
            
        if (i+1) % 20 == 0:
            log.info(f"Processed {i+1}/{len(unique_experiments)}")

    if not results:
        log.error("No configurations placed at least 10 bets.")
        return

    df_res = pd.DataFrame(results)
    
    # Sort and Display Top 10 by ROI
    print("\n" + "="*80)
    print(" TOP 10 STRATEGIES BY ROI (%)")
    print("="*80)
    df_roi = df_res.sort_values(by='roi', ascending=False).head(10)
    print(df_roi.to_string(index=False, float_format="%.2f"))

    # Sort and Display Top 10 by Sharpe Ratio
    print("\n" + "="*80)
    print(" TOP 10 STRATEGIES BY SHARPE RATIO")
    print("="*80)
    df_sharpe = df_res.sort_values(by='sharpe', ascending=False).head(10)
    print(df_sharpe.to_string(index=False, float_format="%.2f"))

    import os
    os.makedirs("results", exist_ok=True)
    df_res.to_csv("results/strategy_sweep.csv", index=False)
    log.info("\nFull results saved to results/strategy_sweep.csv")

if __name__ == "__main__":
    run_sweep()
