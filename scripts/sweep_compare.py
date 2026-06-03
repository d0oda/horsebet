import pandas as pd
from models.backtest import Backtester, BacktestConfig

def run_sweep(df, model_name):
    ev_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
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
                bt.print_report = lambda x: None
                res = bt.run(df)
                roi = (res.total_profit / res.total_staked) * 100 if res.total_bets > 0 else 0.0
                
                results.append({
                    "Model": model_name,
                    "Strategy": "Kelly",
                    "EV": ev,
                    "Kelly": k,
                    "Max_Odds": max_odds,
                    "Bets": res.total_bets,
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
            roi = (res.total_profit / res.total_staked) * 100 if res.total_bets > 0 else 0.0
            
            results.append({
                "Model": model_name,
                "Strategy": "Flat",
                "EV": ev,
                "Kelly": 0.0,
                "Max_Odds": max_odds,
                "Bets": res.total_bets,
                "Profit": res.total_profit,
                "ROI": roi
            })

    return results

def main():
    print("Loading Standard Model predictions...")
    std_df = pd.read_parquet("data/test_preds_standard.parquet")

    results = []
    print("Sweeping Standard Model...")
    results.extend(run_sweep(std_df, "Standard"))

    res_df = pd.DataFrame(results)
    
    # Filter for significance
    res_df = res_df[res_df["Bets"] >= 50]
    
    print("\n=== TOP 20 CONFIGURATIONS BY ROI (Min 50 bets) ===")
    print(res_df.sort_values("ROI", ascending=False).head(20).to_string(index=False))

    print("\n=== TOP 20 CONFIGURATIONS BY PROFIT (Min 50 bets) ===")
    print(res_df.sort_values("Profit", ascending=False).head(20).to_string(index=False))

if __name__ == "__main__":
    main()
