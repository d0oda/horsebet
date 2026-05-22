import pandas as pd
import numpy as np
from models.backtest import Backtester, BacktestConfig

def run_sweep():
    print("Loading OOS predictions from data/oos_preds.parquet...")
    df = pd.read_parquet("data/oos_preds.parquet")
    print(f"Loaded {len(df)} predictions.")

    ev_thresholds = [0.05, 0.10, 0.15, 0.20]
    kelly_fractions = [0.10, 0.25, 0.50]
    max_odds_limits = [15.0, 30.0, 50.0, 100.0]

    results = []

    # 1. Sweep Kelly
    for ev in ev_thresholds:
        for k in kelly_fractions:
            for max_odds in max_odds_limits:
                config = BacktestConfig(
                    ev_threshold=ev,
                    use_kelly=True,
                    kelly_fraction=k,
                    max_odds=max_odds,
                )
                bt = Backtester(config)
                # Suppress printing
                bt.print_report = lambda x: None
                res = bt.run(df)
                if res.total_bets > 0:
                    roi = (res.total_profit / res.total_staked) * 100
                else:
                    roi = 0.0
                
                results.append({
                    "Strategy": "Kelly",
                    "EV": ev,
                    "Kelly_Frac": k,
                    "Max_Odds": max_odds,
                    "Bets": res.total_bets,
                    "Hit_Rate": res.hit_rate,
                    "Profit": res.total_profit,
                    "ROI": roi
                })

    # 2. Sweep Flat Betting
    for ev in ev_thresholds:
        for max_odds in max_odds_limits:
            config = BacktestConfig(
                ev_threshold=ev,
                use_kelly=False,
                flat_stake=1000,
                max_odds=max_odds,
            )
            bt = Backtester(config)
            bt.print_report = lambda x: None
            res = bt.run(df)
            if res.total_bets > 0:
                roi = (res.total_profit / res.total_staked) * 100
            else:
                roi = 0.0
            
            results.append({
                "Strategy": "Flat",
                "EV": ev,
                "Kelly_Frac": 0.0,
                "Max_Odds": max_odds,
                "Bets": res.total_bets,
                "Hit_Rate": res.hit_rate,
                "Profit": res.total_profit,
                "ROI": roi
            })

    res_df = pd.DataFrame(results)
    
    # Filter out combinations with too few bets (noise)
    res_df = res_df[res_df["Bets"] > 50]
    
    print("\n--- TOP 10 BY PURE ROI ---")
    print(res_df.sort_values("ROI", ascending=False).head(10).to_string(index=False))

    print("\n--- TOP 10 BY TOTAL PROFIT ---")
    print(res_df.sort_values("Profit", ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    run_sweep()
