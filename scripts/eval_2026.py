import pandas as pd
import sqlite3

try:
    df = pd.read_parquet("data/oos_preds.parquet")
except Exception as e:
    print(f"Failed to load oos_preds.parquet: {e}")
    exit(1)

# Filter for 2026 races
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'].dt.year == 2026].copy()

# Ensure we have odds_win
if 'odds_win' not in df.columns:
    print("Loading odds from database...")
    conn = sqlite3.connect("horsebet.db")
    odds_df = pd.read_sql("SELECT id as entry_id, odds_win FROM entries WHERE odds_win IS NOT NULL AND odds_win > 0", conn)
    df = df.merge(odds_df, on="entry_id", how="inner")
else:
    df = df[df["odds_win"].notna() & (df["odds_win"] > 0)]

# Calculate EV
df["ev"] = df["win_prob"] * df["odds_win"] - 1.0

# Apply High Profit Config
# EV > 0.25, Min Odds 2.0, Max Odds 20.0
bets = df[(df["ev"] > 0.25) & (df["odds_win"] >= 2.0) & (df["odds_win"] <= 20.0)].copy()

# Sort by date
bets = bets.sort_values(by=["date", "race_id"])

n_bets = len(bets)
wins = bets[bets["finish_pos"] == 1]
n_wins = len(wins)
hit_rate = n_wins / n_bets if n_bets > 0 else 0

total_staked = n_bets * 1000
total_payout = (wins["odds_win"] * 1000).sum()
profit = total_payout - total_staked
roi = profit / total_staked if total_staked > 0 else 0

print("============================================================")
print("  UmaEdge — 2026 YTD Performance (High Profit Config)")
print("============================================================")
print(f"  Bets placed:           {n_bets}")
print(f"  Winning bets:          {n_wins}")
print(f"  Hit rate:              {hit_rate:.1%}")
print(f"  Avg odds:              {bets['odds_win'].mean():.1f}x")
print("-" * 60)
print(f"  Total staked:       ¥ {total_staked:,.0f}")
print(f"  Total payout:       ¥ {total_payout:,.0f}")
print(f"  Total profit:       ¥ {profit:,.0f}")
print(f"  ROI:                   {roi:+.1%}")
print("============================================================\n")

print("  Last 20 bets of 2026:")
print("  Date         Horse                P(win)  Odds     EV   P&L")
print("  -------------------------------------------------------------")

for _, row in bets.tail(20).iterrows():
    won = row["finish_pos"] == 1
    pnl = (row["odds_win"] * 1000) - 1000 if won else -1000
    icon = "✅" if won else "❌"
    
    horse_name = str(row['horse_name']).ljust(20)
    # Handle wide characters spacing roughly
    display_len = sum(2 if ord(c) > 127 else 1 for c in horse_name.strip())
    pad = max(0, 20 - display_len)
    horse_name = horse_name.strip() + " " * pad
    
    print(f"  {row['date'].strftime('%Y-%m-%d')}   {horse_name} {row['win_prob']:5.1%}  {row['odds_win']:4.1f}  {row['ev']:+4.2f}  {pnl:+5.0f} {icon}")

