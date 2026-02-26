#!/usr/bin/env bash
# UmaEdge — Sprint 7.1: Expand Data to 2019–2021
#
# Scrapes 2019, 2020, 2021 JRA races to expand the training dataset
# from ~1,600 to ~3,100+ races. Then optionally triggers a retrain.
#
# Usage:
#   ./scripts/expand_data.sh                    # Scrape all 3 years
#   ./scripts/expand_data.sh --max-races 200    # Limit per year
#   ./scripts/expand_data.sh --retrain          # Scrape + retrain
#   ./scripts/expand_data.sh --dry-run          # Preview only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

MAX_RACES="${1:-1000}"
RETRAIN=false
DRY_RUN=false

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-races)
            MAX_RACES="$2"
            shift 2
            ;;
        --retrain)
            RETRAIN=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Verify venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "⚠️  Python venv not found at: $VENV_PYTHON"
    echo "   Run: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "=============================================="
echo "  UmaEdge — Sprint 7.1: Expand Data"
echo "=============================================="
echo "  Years: 2019, 2020, 2021"
echo "  Max races per year: $MAX_RACES"
echo "  Retrain after: $RETRAIN"
echo "  Dry run: $DRY_RUN"
echo "=============================================="
echo ""

DRY_FLAG=""
if $DRY_RUN; then
    DRY_FLAG="--dry-run"
fi

# Scrape 2019–2021
cd "$PROJECT_DIR"
$VENV_PYTHON -m scraper.batch_scrape \
    --years 2019,2020,2021 \
    --max-races "$MAX_RACES" \
    $DRY_FLAG

# Optional retrain
if $RETRAIN && ! $DRY_RUN; then
    echo ""
    echo "=============================================="
    echo "  Triggering retrain with expanded data..."
    echo "=============================================="
    $VENV_PYTHON -m models.retrain --skip-scrape --calibration isotonic
fi

echo ""
echo "✅ Data expansion complete."
echo "   Next: run 'python -m models.run_evaluation' to verify."
echo ""
