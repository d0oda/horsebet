---
description: How to look up predictions from the predictions JSON file
---

# Predictions JSON Structure

File: `results/predictions_YYYY-MM-DD.json`

## Top-level keys
```json
{
  "model": "2026_v2",
  "date": "2026-03-01",
  "filters": {...},
  "summary": {...},
  "predictions": [...]  // flat list of entries (NOT grouped by race)
}
```

## Entry keys (each item in `predictions`)
```
race_id            → DB race ID (int), e.g. 5503
entry_id           → DB entry ID (int)
horse_name         → Japanese name, e.g. "アトラステソーロ"
fundamental_prob   → Model's fundamental probability (float)
market_model_prob  → Market-adjusted probability (float)
combined_prob      → Final blended probability (float) ← USE THIS
odds               → Current win odds (float) ← NOT "odds_win"
market_prob        → Implied market probability (float)
ev                 → Expected value as decimal (float), e.g. 0.238 = +23.8%
kelly_fraction     → Kelly criterion stake fraction
recommended_stake  → Suggested bet size
is_value_bet       → Boolean, true if EV > threshold
```

## How to look up a specific race

1. Get the DB race ID from the `horsebet.races` table:
```python
from scraper.db import get_session
from sqlalchemy import text

with get_session() as s:
    rows = s.execute(text("""
        SELECT r.id, r.race_number, c.name as course
        FROM horsebet.races r
        JOIN horsebet.courses c ON r.course_id = c.id
        WHERE r.date = '2026-03-01'
        ORDER BY c.name, r.race_number
    """)).fetchall()
```

2. Filter predictions by `race_id`:
```python
import json
with open('results/predictions_2026-03-01.json') as f:
    preds = json.load(f)

race_entries = [e for e in preds['predictions'] if e['race_id'] == 5503]
for e in sorted(race_entries, key=lambda x: x['combined_prob'], reverse=True):
    print(f"{e['horse_name']:<18} {e['combined_prob']:>5.1%} {e['odds']:>5.1f}x {e['ev']:>+6.1%} {'BET' if e['is_value_bet'] else ''}")
```

## Key gotchas
- Entries are a **flat list**, not grouped by race — filter by `race_id`
- Odds key is `odds`, NOT `odds_win`
- Probability key is `combined_prob`, NOT `predicted_prob`
- `race_id` is the DB integer ID, NOT the netkeiba race ID string
- `post_position` is NOT in the predictions — use `entry_id` to join with DB if needed
