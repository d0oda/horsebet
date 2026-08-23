---
description: How to query the UmaEdge database
---

# Database Access

All UmaEdge data lives in a **local SQLite** database file: `horsebet.db`.

## Connection

Set in `.env`:
```
DATABASE_URL=sqlite:///horsebet.db
```

`scraper/db.py` loads this automatically via `python-dotenv`. All Python code uses `get_session()` from there — no schema prefix needed.

## When using Python / SQLAlchemy

```python
from scraper.db import get_session
from sqlalchemy import text

with get_session() as s:
    rows = s.execute(text("SELECT * FROM races WHERE date = :d"), {"d": "2026-08-22"}).fetchall()
```

## Tables

Core tables: `races`, `entries`, `horses`, `jockeys`, `trainers`, `courses`, `results`, `odds_snapshots`

## Common queries

```sql
-- All races on a date
SELECT * FROM races WHERE date = '2026-08-22' ORDER BY course_id, race_number;

-- Entries with odds and results
SELECT e.id, h.name_jp, e.odds_win, r.finish_pos
FROM entries e
JOIN horses h ON h.id = e.horse_id
LEFT JOIN results r ON r.entry_id = e.id
WHERE e.race_id = 5503;
```

## Notes
- Never use `horsebet.` schema prefix in SQL — it's plain SQLite, not Postgres
- Table names are unqualified: `races`, `entries`, etc.
