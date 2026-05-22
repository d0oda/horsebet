import pandas as pd
from models.backtest import Backtester, BacktestConfig

print("Running 2026 Walk-Forward Backtest (Model: retrain_20260522_2234)")
df = pd.read_parquet("data/oos_preds.parquet")

# Filter for 2026 races
df_2026 = df[df["date"] >= "2026-01-01"].copy()
print(f"Loaded {len(df_2026)} predictions for 2026.")

# Configure optimized flat betting strategy
config = BacktestConfig(
    ev_threshold=0.20,
    max_odds=30.0,
    use_kelly=False,
    flat_stake=1000
)

bt = Backtester(config)
res = bt.run(df_2026)
bt.print_report(res)
