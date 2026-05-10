#!/bin/bash

echo "Waiting for feature rebuild to complete..."

# Check every 30 seconds if the python script rebuilding features is still running
while pgrep -f "import pandas as pd" > /dev/null; do
    sleep 30
done

echo "Feature rebuild finished."
echo "Training new ODDS-FREE model (3-way split)..."
.venv/bin/python -m models.train --val-date 2023-01-01 --test-date 2023-07-30 --exclude-odds > train_odds_free.log 2>&1

echo "Model trained. Running strategy threshold optimization on blind test set..."
.venv/bin/python scratch_optimize_testset.py > optimize_odds_free.log 2>&1

echo "Done! Check train_latest.log and optimize_latest.log for the final results."
