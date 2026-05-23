#!/bin/bash
for date in 2026-04-04 2026-04-05 2026-04-11; do
  PYTHONPATH=. ./.venv/bin/python pipeline.py --date $date --version retrain_20260522_2234 --ev-threshold 0.2 --skip-scrape --skip-odds
done
