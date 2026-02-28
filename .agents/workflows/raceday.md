---
description: Run the full race day pipeline (scrape → odds → predict → frontend)
---

# Race Day Pipeline

// turbo-all

## Full Pipeline (new race day)
```bash
python pipeline.py --date YYYY-MM-DD
```

## Re-run with existing data (skip scraping)
```bash
python pipeline.py --date YYYY-MM-DD --skip-scrape
```

## Update odds only and re-predict
```bash
python pipeline.py --date YYYY-MM-DD --skip-scrape
```

## Re-build frontend only (predictions already saved)
```bash
python pipeline.py --date YYYY-MM-DD --skip-scrape --skip-odds --skip-predict
```

## Custom thresholds
```bash
python pipeline.py --date YYYY-MM-DD --ev-threshold 0.08 --max-odds 20
```

## Serve the results page
```bash
cd results && python -m http.server 8080
```
