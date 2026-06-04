#!/bin/bash
export PYTHONPATH=.

echo "=== Evaluating retrain_20260523_2115 ==="
./.venv/bin/python fast_backtest.py retrain_20260523_2115 0.30
./.venv/bin/python scripts/analyze_april.py > retrain_results.txt
cat retrain_results.txt | grep -A 6 "==="

echo "=== Evaluating 2026_v2 (Hybrid) ==="
./.venv/bin/python fast_backtest.py 2026_v2 0.30
./.venv/bin/python scripts/analyze_april.py > v2_results.txt
cat v2_results.txt | grep -A 6 "==="
