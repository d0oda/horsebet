import json
from pathlib import Path
from sqlalchemy import text
from scraper.db import get_session

dates = [
    '2026-04-04', '2026-04-05', '2026-04-11', '2026-04-12', 
    '2026-04-18', '2026-04-19', '2026-04-25', '2026-04-26'
]

total_bets = 0
won_bets = 0
total_staked = 0
total_returned = 0

with get_session() as session:
    res = session.execute(text("SELECT id, race_id FROM entries WHERE finish_pos = 1")).fetchall()
    winning_entries = {r[0] for r in res}

print("WINNING BETS:")
for d in dates:
    path = Path(f"results/predictions_{d}.json")
    if not path.exists():
        continue
    with open(path) as f:
        data = json.load(f)
    for p in data.get("predictions", []):
        if p.get("is_value_bet") or p.get("is_value"):
            stake = p.get("recommended_stake", 1000)
            if stake <= 0:
                continue
            
            total_bets += 1
            total_staked += stake
            
            if p["entry_id"] in winning_entries:
                won_bets += 1
                total_returned += stake * p["odds"]
                print(f"[{d}] Race {p['race_id']} - Entry {p['entry_id']} | Odds: {p['odds']} | Return: {stake * p['odds']}")

print("\n==================================")
print(f"Total Bets:   {total_bets}")
print(f"Total Won:    {won_bets}")
print(f"Total Staked: ¥{total_staked:,}")
print(f"Total Return: ¥{int(total_returned):,}")
print(f"Profit:       ¥{int(total_returned - total_staked):,}")
print("==================================\n")
