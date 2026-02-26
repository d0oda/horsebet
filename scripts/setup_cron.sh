#!/usr/bin/env bash
# UmaEdge — Cron Setup for Sprint 6
#
# Installs weekly retrain + drift check cron jobs.
#
# Usage:
#   ./scripts/setup_cron.sh              # Preview cron entries
#   ./scripts/setup_cron.sh --install    # Install to crontab

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

# Verify venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "⚠️  Python venv not found at: $VENV_PYTHON"
    echo "   Run: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Cron entries
RETRAIN_CRON="0 6 * * 1 cd $PROJECT_DIR && $VENV_PYTHON -m models.retrain --calibration isotonic >> $PROJECT_DIR/data/retrain.log 2>&1"
DRIFT_CRON="0 9 * * 3 cd $PROJECT_DIR && $VENV_PYTHON -m models.drift >> $PROJECT_DIR/data/drift.log 2>&1"

echo ""
echo "=============================================="
echo "  UmaEdge — Sprint 6 Cron Setup"
echo "=============================================="
echo ""
echo "  Retrain (Mondays 6am):"
echo "    $RETRAIN_CRON"
echo ""
echo "  Drift check (Wednesdays 9am):"
echo "    $DRIFT_CRON"
echo ""

if [ "${1:-}" = "--install" ]; then
    echo "Installing to crontab..."

    # Ensure data dir exists for logs
    mkdir -p "$PROJECT_DIR/data"

    # Add to crontab (preserving existing entries)
    (crontab -l 2>/dev/null || true; echo "$RETRAIN_CRON"; echo "$DRIFT_CRON") | sort -u | crontab -

    echo "✅ Cron jobs installed. Verify with: crontab -l"
else
    echo "  To install, run:"
    echo "    ./scripts/setup_cron.sh --install"
fi
echo ""
