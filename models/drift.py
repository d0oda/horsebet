"""
UmaEdge — Model Drift Tracker (Sprint 4.2).

Tracks model performance metrics over time:
  - AUC, log-loss, Brier score per version
  - Stores in data/drift_log.json
  - Provides summary for frontend dashboard

Usage:
    from models.drift import log_metrics, get_drift_summary
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("drift")

DRIFT_LOG_PATH = Path(__file__).parent.parent / "data" / "drift_log.json"


def _load_log() -> list[dict]:
    """Load the drift log file."""
    if DRIFT_LOG_PATH.exists():
        try:
            return json.loads(DRIFT_LOG_PATH.read_text())
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_log(entries: list[dict]):
    """Save the drift log file."""
    DRIFT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRIFT_LOG_PATH.write_text(json.dumps(entries, indent=2, default=str))


def log_metrics(
    version: str,
    auc: float,
    log_loss: float,
    brier: float = 0.0,
    n_samples: int = 0,
    calibration_error: float = 0.0,
    notes: str = "",
):
    """
    Log a set of model performance metrics.

    Args:
        version: Model version identifier
        auc: Area under ROC curve
        log_loss: Log-loss (negative log-likelihood)
        brier: Brier score
        n_samples: Number of validation samples
        calibration_error: Average calibration error
        notes: Optional notes
    """
    entries = _load_log()

    entry = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "auc": round(auc, 6),
        "log_loss": round(log_loss, 6),
        "brier": round(brier, 6),
        "n_samples": n_samples,
        "calibration_error": round(calibration_error, 6),
        "notes": notes,
    }

    entries.append(entry)
    _save_log(entries)

    log.info(f"📊 Logged metrics for {version}: AUC={auc:.4f}, log-loss={log_loss:.4f}")


def get_drift_summary() -> dict:
    """
    Get a summary of model drift over time.

    Returns:
        Dict with latest metrics, trends, and full history.
    """
    entries = _load_log()
    if not entries:
        return {"status": "no_data", "entries": []}

    latest = entries[-1]

    # Compute trends (if 2+ entries)
    trend = {}
    if len(entries) >= 2:
        prev = entries[-2]
        trend["auc_change"] = latest["auc"] - prev["auc"]
        trend["log_loss_change"] = latest["log_loss"] - prev["log_loss"]
        trend["direction"] = "improving" if trend["auc_change"] > 0 else "degrading"

    # Compute averages
    auc_values = [e["auc"] for e in entries if e.get("auc")]
    ll_values = [e["log_loss"] for e in entries if e.get("log_loss")]

    return {
        "status": "ok",
        "n_versions": len(entries),
        "latest": latest,
        "trend": trend,
        "avg_auc": sum(auc_values) / len(auc_values) if auc_values else 0,
        "avg_log_loss": sum(ll_values) / len(ll_values) if ll_values else 0,
        "entries": entries[-20:],  # last 20 for dashboard
    }


def print_drift_report():
    """Print a formatted drift report."""
    summary = get_drift_summary()
    if summary["status"] == "no_data":
        print("No drift data recorded yet. Run retrain.py first.")
        return

    print(f"\n{'=' * 60}")
    print("  UmaEdge — Model Drift Report")
    print(f"{'=' * 60}")
    print(f"  Total versions tracked:  {summary['n_versions']}")
    print(f"  Average AUC:             {summary['avg_auc']:.4f}")
    print(f"  Average Log-loss:        {summary['avg_log_loss']:.4f}")

    if summary.get("trend"):
        t = summary["trend"]
        direction = "📈" if t["direction"] == "improving" else "📉"
        print(f"\n  Latest trend: {direction} {t['direction']}")
        print(f"    AUC change:      {t['auc_change']:+.4f}")
        print(f"    Log-loss change: {t['log_loss_change']:+.4f}")

    print(f"\n  {'Version':<25} {'AUC':>7} {'LogLoss':>8} {'Samples':>8}")
    print(f"  {'-' * 50}")
    for e in summary["entries"][-10:]:
        print(f"  {e['version']:<25} {e['auc']:>7.4f} {e['log_loss']:>8.4f} {e['n_samples']:>8}")

    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_drift_report()
