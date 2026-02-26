"""
UmaEdge — Sprint 6: Paper Trading Orchestrator.

Manages paper trading validation and production-readiness:
  6.1 — Place paper bets on upcoming races
  6.2 — Reconcile pending paper trades against actual results
  6.3 — Run weekly retrain pipeline
  6.4 — Check model drift

Usage:
    # Place paper bets on specific races
    python -m models.run_sprint6 --paper-trade 202506010101 202506010102

    # Reconcile pending paper trades
    python -m models.run_sprint6 --reconcile

    # Run weekly retrain
    python -m models.run_sprint6 --retrain

    # Check model drift
    python -m models.run_sprint6 --drift

    # Full weekend workflow: retrain → paper-trade → reconcile
    python -m models.run_sprint6 --weekend 202506010101 202506010102

    # Print summary of all paper trading activity
    python -m models.run_sprint6 --summary
"""

import argparse
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sprint6")


def banner(title: str):
    print(f"\n{'#' * 70}")
    print(f"#  {title}")
    print(f"{'#' * 70}\n")


# ---------------------------------------------------------------------------
# 6.1 — Paper Trading
# ---------------------------------------------------------------------------

def do_paper_trade(
    race_ids: list[str],
    ev_threshold: float = 0.12,
    flat_stake: int = 1000,
):
    """Place paper bets on the given races."""
    banner("Sprint 6.1 — Paper Trading")

    from models.paper_trade import PaperTrader

    trader = PaperTrader()
    total_bets = 0

    for race_id in race_ids:
        log.info(f"Processing race {race_id}...")
        try:
            # Convert to int if it's a netkeiba ID string
            rid = int(race_id) if race_id.isdigit() else race_id
            bets = trader.place_paper_bets(
                race_id=rid,
                ev_threshold=ev_threshold,
                flat_stake=flat_stake,
            )
            total_bets += len(bets)
            log.info(f"  Placed {len(bets)} paper bets for race {race_id}")
        except Exception as e:
            log.error(f"  Failed for race {race_id}: {e}")

    log.info(f"\n✅ Total paper bets placed: {total_bets}")
    trader.print_summary()


# ---------------------------------------------------------------------------
# 6.2 — Reconcile
# ---------------------------------------------------------------------------

def do_reconcile():
    """Reconcile pending paper trades against actual results."""
    banner("Sprint 6.2 — Reconcile Paper Trades")

    from models.paper_trade import PaperTrader

    trader = PaperTrader()
    summary = trader.reconcile()
    trader.print_summary()
    return summary


# ---------------------------------------------------------------------------
# 6.3 — Weekly Retrain
# ---------------------------------------------------------------------------

def do_retrain(
    calibration: str = "isotonic",
    skip_scrape: bool = False,
):
    """Run the weekly retrain pipeline."""
    banner("Sprint 6.3 — Weekly Retrain")

    from models.retrain import run_retrain

    metrics = run_retrain(
        calibration_method=calibration,
        skip_scrape=skip_scrape,
    )
    return metrics


# ---------------------------------------------------------------------------
# 6.4 — Drift Check
# ---------------------------------------------------------------------------

def do_drift():
    """Check model drift and print report."""
    banner("Sprint 6.4 — Model Drift Check")

    from models.drift import print_drift_report
    print_drift_report()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def do_summary():
    """Print comprehensive paper trading summary."""
    banner("Paper Trading Summary")

    from models.paper_trade import PaperTrader
    trader = PaperTrader()
    trader.print_summary()


# ---------------------------------------------------------------------------
# 7.5 — Live Trading
# ---------------------------------------------------------------------------

def do_live_trade(
    race_ids: list[str],
    ev_threshold: float = 0.12,
    flat_stake: int = 100,
    confirm: bool = False,
):
    """Place REAL bets on the given races."""
    banner("Sprint 7.5 — 🔴 LIVE Trading")

    from models.live_trade import LiveTrader

    trader = LiveTrader(stake=flat_stake, ev_threshold=ev_threshold)
    total_bets = 0

    for race_id in race_ids:
        log.info(f"Processing race {race_id}...")
        try:
            rid = int(race_id) if race_id.isdigit() else race_id
            bets = trader.place_live_bets(
                race_id=rid,
                confirm=confirm,
            )
            total_bets += len(bets)
            log.info(f"  🔴 Placed {len(bets)} LIVE bets for race {race_id}")
        except Exception as e:
            log.error(f"  Failed for race {race_id}: {e}")

    log.info(f"\n🔴 Total LIVE bets placed: {total_bets}")
    trader.print_summary()


def do_live_summary():
    """Print live trading summary."""
    banner("🔴 Live Trading Summary")

    from models.live_trade import LiveTrader
    trader = LiveTrader()
    trader.print_summary()


def do_live_reconcile():
    """Reconcile pending live trades."""
    banner("Sprint 7.5 — Reconcile Live Trades")

    from models.live_trade import LiveTrader
    trader = LiveTrader()
    trader.reconcile()
    trader.print_summary()


# ---------------------------------------------------------------------------
# Weekend Workflow
# ---------------------------------------------------------------------------

def do_weekend(
    race_ids: list[str],
    ev_threshold: float = 0.12,
    flat_stake: int = 1000,
    calibration: str = "isotonic",
):
    """Full weekend workflow: retrain → paper-trade → reconcile old."""
    banner("Sprint 6 — Full Weekend Workflow")

    # Step 1: Reconcile any pending trades from last week
    log.info("=== Step 1/3: Reconcile previous trades ===")
    do_reconcile()

    # Step 2: Retrain with latest data
    log.info("\n=== Step 2/3: Weekly retrain ===")
    do_retrain(calibration=calibration, skip_scrape=False)

    # Step 3: Place new paper bets
    log.info("\n=== Step 3/3: Place paper bets ===")
    do_paper_trade(race_ids, ev_threshold=ev_threshold, flat_stake=flat_stake)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UmaEdge — Sprint 6: Paper Trading Orchestrator"
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--paper-trade", nargs="+", metavar="RACE_ID",
        help="Place paper bets on the given race IDs (12-digit netkeiba IDs)",
    )
    mode.add_argument(
        "--reconcile", action="store_true",
        help="Reconcile pending paper trades against actual results",
    )
    mode.add_argument(
        "--retrain", action="store_true",
        help="Run the weekly retrain pipeline",
    )
    mode.add_argument(
        "--drift", action="store_true",
        help="Check model drift and print report",
    )
    mode.add_argument(
        "--summary", action="store_true",
        help="Print paper trading summary",
    )
    mode.add_argument(
        "--weekend", nargs="+", metavar="RACE_ID",
        help="Full weekend workflow: reconcile → retrain → paper-trade",
    )
    mode.add_argument(
        "--live", nargs="+", metavar="RACE_ID",
        help="🔴 Place REAL bets (requires --confirm)",
    )
    mode.add_argument(
        "--live-summary", action="store_true",
        help="Print live trading summary",
    )
    mode.add_argument(
        "--live-reconcile", action="store_true",
        help="Reconcile pending live trades",
    )

    # Options
    parser.add_argument(
        "--ev-threshold", type=float, default=0.12,
        help="EV threshold for paper bets (default: 0.12 = 12%%)",
    )
    parser.add_argument(
        "--flat-stake", type=int, default=1000,
        help="Hypothetical stake per bet in yen (default: 1000)",
    )
    parser.add_argument(
        "--calibration", type=str, default="isotonic",
        choices=["none", "platt", "isotonic"],
        help="Calibration method for retrain (default: isotonic)",
    )
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip scraping in retrain step",
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Confirm live trading (required for --live)",
    )

    args = parser.parse_args()

    start = datetime.now()

    if args.paper_trade:
        do_paper_trade(args.paper_trade, args.ev_threshold, args.flat_stake)
    elif args.reconcile:
        do_reconcile()
    elif args.retrain:
        do_retrain(args.calibration, args.skip_scrape)
    elif args.drift:
        do_drift()
    elif args.summary:
        do_summary()
    elif args.weekend:
        do_weekend(
            args.weekend,
            ev_threshold=args.ev_threshold,
            flat_stake=args.flat_stake,
            calibration=args.calibration,
        )
    elif args.live:
        do_live_trade(
            args.live,
            ev_threshold=args.ev_threshold,
            flat_stake=args.flat_stake,
            confirm=args.confirm,
        )
    elif args.live_summary:
        do_live_summary()
    elif args.live_reconcile:
        do_live_reconcile()

    elapsed = datetime.now() - start
    log.info(f"\nDone in {elapsed.total_seconds():.1f}s")


if __name__ == "__main__":
    main()
