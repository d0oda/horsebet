---
description: How to query the UmaEdge database
---

# Database Access

All UmaEdge tables live in the **`horsebet` schema** on Supabase, NOT `public`.

## When using Supabase MCP tools (`execute_sql`, `apply_migration`, etc.)

Always qualify table names:
```sql
-- ✅ Correct
SELECT * FROM horsebet.races;
SELECT * FROM horsebet.entries;
SELECT * FROM horsebet.horses;

-- ❌ Wrong (will fail with "relation does not exist")
SELECT * FROM races;
```

## Project details

- **Project ID**: `utadojvegaohlgqchsdy`
- **Region**: `eu-west-1`
- **Schema**: `horsebet`

## Tables

Core tables: `races`, `entries`, `horses`, `jockeys`, `trainers`, `courses`, `results`, `odds_snapshots`

## When using Python / SQLAlchemy

The `scraper/db.py` module sets `search_path TO horsebet, public` automatically on every connection, so Python code does NOT need to qualify table names.
