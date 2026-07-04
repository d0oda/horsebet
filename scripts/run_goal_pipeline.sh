#!/bin/bash
set -e
echo "Starting Pipeline..."

echo "============================================="
echo " Phase 1: Rebuilding ALL features (2014-2026)"
echo "============================================="
PYTHONPATH=. .venv/bin/python scripts/rebuild_all_features.py

echo "============================================="
echo " Phase 2: Training LightGBM & XGBoost Models "
echo "============================================="
PYTHONPATH=. .venv/bin/python models/train.py

echo "============================================="
echo " Phase 3: Generating Test Predictions        "
echo "============================================="
PYTHONPATH=. .venv/bin/python scripts/generate_test_preds.py

echo "============================================="
echo " Pipeline Complete! 🎉                       "
echo "============================================="
