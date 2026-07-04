import pandas as pd
import numpy as np
import sqlite3

# Load out-of-sample predictions
try:
    df = pd.read_parquet("data/oos_preds.parquet")
except Exception as e:
    print(f"Failed to load oos_preds.parquet: {e}")
    exit(1)

# We need odds_win and finish_pos. oos_preds.parquet only has race_id, entry_id, date, horse_name, finish_pos, win_prob
# Wait, let's check columns
print("Columns in oos_preds:", list(df.columns))

# Drop missing odds
df = df[df["odds_win"].notna() & (df["odds_win"] > 0)].copy()

# Calculate EV: P(win) * Odds - 1
df["ev"] = df["win_prob"] * df["odds_win"] - 1.0

print(f"Total out-of-sample races: {df['race_id'].nunique()}")
print(f"Total entries: {len(df)}")
print("-" * 50)

results = []

ev_thresholds = [0.0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
max_odds_list = [10, 15, 20, 30, 50, 999]
min_odds_list = [1.0, 2.0, 3.0, 5.0]

for ev_t in ev_thresholds:
    for max_o in max_odds_list:
        for min_o in min_odds_list:
            
            bets = df[(df["ev"] > ev_t) & (df["odds_win"] <= max_o) & (df["odds_win"] >= min_o)]
            n_bets = len(bets)
            
            if n_bets < 50:
                continue
                
            wins = bets[bets["finish_pos"] == 1]
            n_wins = len(wins)
            hit_rate = n_wins / n_bets if n_bets > 0 else 0
            
            # Flat stake 1000
            total_staked = n_bets * 1000
            total_payout = (wins["odds_win"] * 1000).sum()
            profit = total_payout - total_staked
            roi = profit / total_staked if total_staked > 0 else 0
            
            results.append({
                "ev_threshold": ev_t,
                "min_odds": min_o,
                "max_odds": max_o,
                "bets": n_bets,
                "hit_rate": hit_rate,
                "roi": roi,
                "profit": profit
            })

results_df = pd.DataFrame(results)
if results_df.empty:
    print("No configurations met the criteria.")
    exit(0)

best_roi = results_df[results_df["bets"] >= 200].sort_values("roi", ascending=False).head(10)

print("\nTOP 10 CONFIGURATIONS BY ROI (Min 200 bets):")
print(best_roi.to_string(index=False, formatters={
    'roi': '{:.1%}'.format, 
    'hit_rate': '{:.1%}'.format,
    'profit': '¥{:,.0f}'.format
}))

best_profit = results_df[results_df["bets"] >= 200].sort_values("profit", ascending=False).head(10)
print("\nTOP 10 CONFIGURATIONS BY TOTAL PROFIT (Min 200 bets):")
print(best_profit.to_string(index=False, formatters={
    'roi': '{:.1%}'.format, 
    'hit_rate': '{:.1%}'.format,
    'profit': '¥{:,.0f}'.format
}))
