---
description: Run the full race day pipeline (scrape → odds → predict → frontend)
---

# Race Day Pipeline

// turbo-all

## CRITICAL RULES FOR PIPELINE EXECUTION
1. **NEVER use `--skip-odds` for an active/upcoming race day.** If you skip odds, the Expected Value (EV) calculation will be 0% for all horses and no value bets will be found.
2. **ALWAYS use a hybrid model.** Do NOT use standard `retrain_*` models unless you specifically trained a hybrid retrain model. Hybrid models are required for the correct blending of fundamental and market features. Currently using `retrain_20260822_2143`.

## Live Config (Optimal Sharpe — set as pipeline defaults)
- **EV threshold**: ≥ 15% (`--ev-threshold 0.15`)
- **Odds range**: 2.0 – 20.0x (`--min-odds 2.0 --max-odds 20.0`)
- **Stake**: flat ¥1,000 per bet (Kelly cap = 0)
- **Rationale**: Best risk-adjusted returns across 67 2026 race days — Sharpe 1.21, ROI +6.8%, 709 bets

These are now the **pipeline.py defaults** — no flags needed for standard runs.

## Full Pipeline (new race day)
```bash
python pipeline.py --date YYYY-MM-DD --version retrain_20260822_2143
```

## Re-run with existing data (skip scraping)
```bash
python pipeline.py --date YYYY-MM-DD --skip-scrape --version retrain_20260822_2143
```

## Update odds only and re-predict
```bash
python pipeline.py --date YYYY-MM-DD --skip-scrape --version retrain_20260822_2143
```

## Re-build frontend only (predictions already saved)
```bash
python pipeline.py --date YYYY-MM-DD --skip-scrape --skip-odds --skip-predict --version retrain_20260822_2143
```

## Serve the results page
```bash
cd results && python -m http.server 8080
```
