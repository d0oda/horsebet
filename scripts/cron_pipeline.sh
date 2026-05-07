#!/bin/bash
# UmaEdge — Daily Cron Pipeline.
# Scrapes today's races, generates predictions, checks database health,
# and outputs the frontend bundle.

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

echo "=== UmaEdge Daily Pipeline Execution ==="
date

# Load virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Determine target date (default to today JST)
TARGET_DATE=$(TZ='Asia/Tokyo' date +%Y-%m-%d)
echo "Target Date: $TARGET_DATE"

echo "--- 1. Checking Database Health ---"
python scripts/monitor_db_health.py

echo "--- 2. Running Daily Prediction Pipeline ---"
python pipeline.py --date "$TARGET_DATE"

echo "=== Pipeline Completed Successfully ==="
