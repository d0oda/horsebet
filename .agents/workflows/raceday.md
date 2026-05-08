---
description: Run the full race day pipeline (scrape → odds → predict → frontend)
---

# Race Day Pipeline

// turbo-all

## CRITICAL RULES FOR PIPELINE EXECUTION
1. **NEVER use `--skip-odds` for an active/upcoming race day.** If you skip odds, the Expected Value (EV) calculation will be 0% for all horses and no value bets will be found.
2. **ALWAYS use a hybrid model.** The default is `2026_v2`. Do NOT use standard `retrain_*` models unless you specifically trained a hybrid retrain model. Hybrid models are required for the correct blending of fundamental and market features.

## Full Pipeline (new race day)
```bash
python pipeline.py --date YYYY-MM-DD --version 2026_v2
```

## Re-run with existing data (skip scraping)
```bash
python pipeline.py --date YYYY-MM-DD --skip-scrape --version 2026_v2
```

## Update odds only and re-predict
```bash
python pipeline.py --date YYYY-MM-DD --skip-scrape --version 2026_v2
```

## Re-build frontend only (predictions already saved)
```bash
python pipeline.py --date YYYY-MM-DD --skip-scrape --skip-odds --skip-predict --version 2026_v2
```

## Custom thresholds
```bash
python pipeline.py --date YYYY-MM-DD --version 2026_v2 --ev-threshold 0.08 --max-odds 20
```

## Serve the results page
```bash
cd results && python -m http.server 8080
```
