import pandas as pd
from models.backtest import Backtester, BacktestConfig

df = pd.read_parquet("data/oos_preds.parquet")
df_2026 = df[df["date"] >= "2026-01-01"].copy()

config = BacktestConfig(
    ev_threshold=0.20,
    max_odds=30.0,
    use_kelly=False,
    flat_stake=1000
)

bt = Backtester(config)
res = bt.run(df_2026)

df_res = bt.to_dataframe(res)
print("Total bets:", len(df_res))
print("Total won:", len(df_res[df_res["profit"] > 0]))
print("\nTop 10 most profitable bets:")
print(df_res.sort_values("profit", ascending=False).head(10)[["date", "race_id", "horse", "odds", "model_prob", "profit"]])

print("\nP&L over time (by month):")
df_res["month"] = df_res["date"].str[:7]
monthly = df_res.groupby("month").agg(
    bets=("profit", "count"),
    wins=("profit", lambda x: (x > 0).sum()),
    staked=("stake", "sum"),
    profit=("profit", "sum")
)
monthly["ROI"] = (monthly["profit"] / monthly["staked"]) * 100
print(monthly)

