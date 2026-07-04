#!/bin/bash
echo "Waiting for fix_winners_odds.py to finish..."
while true; do
    if grep -q "Repair complete" logs/fix_winners.log; then
        echo "Repair finished! Patching features..."
        PYTHONPATH=. .venv/bin/python scripts/patch_features_odds.py
        
        echo "Generating new predictions..."
        PYTHONPATH=. .venv/bin/python scripts/generate_test_preds.py
        
        echo "Done with everything!"
        break
    fi
    sleep 5
done
