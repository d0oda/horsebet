#!/bin/bash
for date in 2026-04-04 2026-04-05 2026-04-11 2026-04-12 2026-04-18 2026-04-19 2026-04-25 2026-04-26; do
  echo "Running pipeline for $date..."
  PYTHONPATH=. ./.venv/bin/python pipeline.py --date $date --version retrain_20260523_2115 --ev-threshold 0.2 --skip-scrape --skip-odds
done
