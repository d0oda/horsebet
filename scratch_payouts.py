import json

with open("results/data.json") as f:
    data = json.load(f)

print(f"{'Venue':<15} {'Race':<5} {'Horse':<20} {'Odds':<6} {'Return (1000¥)':<15}")
print("-" * 65)

total_return = 0
winners_count = 0

for bet in data.get("value_bets", []):
    if bet.get("is_winner"):
        payout = int(bet["odds"] * 1000)
        total_return += payout
        winners_count += 1
        print(f"{bet['venue']:<15} R{bet['race_number']:<4} {bet['horse_name']:<20} {bet['odds']:<6.1f} ¥{payout:,}")

print("-" * 65)
print(f"Total Winners: {winners_count}")
print(f"Total Returns: ¥{total_return:,}")
print(f"Total Staked (63 bets * 1000): ¥63,000")
print(f"Total Profit: ¥{total_return - 63000:,}")
