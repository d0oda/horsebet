"""
UmaEdge — Paper Trading Mode (Sprint 4.3).

Simulates live trading without real stakes:
  1. For upcoming races, runs predictions
  2. Logs hypothetical bets to data/paper_trades.json
  3. Tracks cumulative paper P&L

Usage:
    python -m models.paper_trade --race-id 123
    python -m models.paper_trade --reconcile
"""

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("paper_trade")

PAPER_TRADES_PATH = Path(__file__).parent.parent / "data" / "paper_trades.json"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class PaperBet:
    """A hypothetical bet placed during paper trading."""
    race_id: int
    entry_id: int
    horse_name: str
    date: str
    model_prob: float
    market_prob: float
    ev: float
    hypothetical_stake: int
    odds: float
    status: str = "pending"  # pending, won, lost
    finish_pos: Optional[int] = None
    payout: int = 0
    profit: int = 0
    created_at: str = ""


# ---------------------------------------------------------------------------
# Paper Trade Manager
# ---------------------------------------------------------------------------

class PaperTrader:
    """Manages paper trading state and reconciliation."""

    def __init__(self):
        self.trades = self._load_trades()

    def _load_trades(self) -> list[dict]:
        if PAPER_TRADES_PATH.exists():
            try:
                return json.loads(PAPER_TRADES_PATH.read_text())
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_trades(self):
        PAPER_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
        PAPER_TRADES_PATH.write_text(json.dumps(self.trades, indent=2, default=str))

    def place_paper_bets(
        self,
        race_id: int,
        ev_threshold: float = 0.10,
        flat_stake: int = 1000,
    ) -> list[PaperBet]:
        """
        Run predictions on a race and log paper bets.

        Args:
            race_id: Database race ID
            ev_threshold: Minimum EV to place a bet
            flat_stake: Hypothetical stake per bet

        Returns:
            List of paper bets placed
        """
        from models.predict import predict_and_store

        log.info(f"Running predictions for race {race_id}...")
        preds_df = predict_and_store(race_id, store_to_db=False, ev_threshold=ev_threshold)

        if preds_df is None or preds_df.empty:
            log.warning(f"No predictions for race {race_id}")
            return []

        bets = []
        for _, row in preds_df.iterrows():
            model_prob = row.get("win_prob", 0)
            odds = row.get("odds_win", 0)

            if not odds or odds <= 0:
                continue

            market_prob = 1.0 / odds
            ev = model_prob - market_prob

            if ev < ev_threshold:
                continue

            bet = PaperBet(
                race_id=race_id,
                entry_id=row.get("entry_id", 0),
                horse_name=row.get("horse_name", "?"),
                date=str(row.get("date", "")),
                model_prob=round(model_prob, 4),
                market_prob=round(market_prob, 4),
                ev=round(ev, 4),
                hypothetical_stake=flat_stake,
                odds=odds,
                created_at=datetime.now().isoformat(),
            )
            bets.append(bet)
            self.trades.append(asdict(bet))

        self._save_trades()
        log.info(f"📝 Placed {len(bets)} paper bets for race {race_id}")

        # Notify
        try:
            from notifications.dispatcher import notify_bet_placed
            for bet in bets:
                notify_bet_placed({
                    "horse_name": bet.horse_name,
                    "model_prob": bet.model_prob,
                    "odds": bet.odds,
                    "ev": bet.ev,
                    "stake": bet.hypothetical_stake,
                    "live": False,
                })
        except Exception:
            pass  # notifications are best-effort

        return bets

    def reconcile(self) -> dict:
        """
        Reconcile pending paper trades against actual results.

        Returns:
            Summary of reconciliation.
        """
        from scraper.db import get_session
        from sqlalchemy import text

        reconciled = 0
        for trade in self.trades:
            if trade["status"] != "pending":
                continue

            with get_session() as session:
                result = session.execute(text("""
                    SELECT res.finish_pos
                    FROM results res
                    JOIN entries e ON e.id = res.entry_id
                    WHERE e.id = :entry_id
                """), {"entry_id": trade["entry_id"]}).fetchone()

                if result:
                    finish_pos = result[0]
                    trade["finish_pos"] = finish_pos
                    won = finish_pos == 1 if finish_pos is not None else False
                    trade["payout"] = int(trade["hypothetical_stake"] * trade["odds"]) if won else 0
                    trade["profit"] = trade["payout"] - trade["hypothetical_stake"]
                    trade["status"] = "won" if won else "lost"
                    reconciled += 1

        self._save_trades()

        # Compute summary
        settled = [t for t in self.trades if t["status"] != "pending"]
        total_staked = sum(t["hypothetical_stake"] for t in settled)
        total_payout = sum(t["payout"] for t in settled)
        total_profit = total_payout - total_staked
        wins = sum(1 for t in settled if t["status"] == "won")

        summary = {
            "reconciled_now": reconciled,
            "total_settled": len(settled),
            "total_pending": sum(1 for t in self.trades if t["status"] == "pending"),
            "wins": wins,
            "total_staked": total_staked,
            "total_payout": total_payout,
            "total_profit": total_profit,
            "roi_pct": (total_profit / total_staked * 100) if total_staked > 0 else 0,
        }

        log.info(f"✅ Reconciled {reconciled} trades")

        # Notify
        try:
            from notifications.dispatcher import notify_reconciliation
            notify_reconciliation(summary)
        except Exception:
            pass  # notifications are best-effort

        return summary

    def print_summary(self):
        """Print paper trading summary."""
        settled = [t for t in self.trades if t["status"] != "pending"]
        pending = [t for t in self.trades if t["status"] == "pending"]

        total_staked = sum(t["hypothetical_stake"] for t in settled)
        total_payout = sum(t["payout"] for t in settled)
        total_profit = total_payout - total_staked
        wins = sum(1 for t in settled if t["status"] == "won")
        hit_rate = (wins / len(settled) * 100) if settled else 0

        print(f"\n{'=' * 60}")
        print("  UmaEdge — Paper Trading Summary")
        print(f"{'=' * 60}")
        print(f"  Total trades:    {len(self.trades):>6}")
        print(f"  Settled:         {len(settled):>6}")
        print(f"  Pending:         {len(pending):>6}")
        print(f"  Wins:            {wins:>6}")
        print(f"  Hit rate:        {hit_rate:>5.1f}%")
        print("-" * 60)
        print(f"  Paper staked:    ¥{total_staked:>10,}")
        print(f"  Paper payout:    ¥{total_payout:>10,}")
        print(f"  Paper profit:    ¥{total_profit:>10,}")
        if total_staked > 0:
            print(f"  Paper ROI:       {total_profit / total_staked * 100:>5.1f}%")
        print(f"{'=' * 60}")

        if pending:
            print(f"\n  Pending bets ({len(pending)}):")
            for t in pending[-5:]:
                print(f"    Race {t['race_id']} | {t['horse_name']:<14} "
                      f"P={t['model_prob']:.1%} odds={t['odds']:.1f}x")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Paper Trading")
    parser.add_argument("--race-id", type=int, help="Place paper bets for this race")
    parser.add_argument("--reconcile", action="store_true", help="Reconcile pending trades")
    parser.add_argument("--summary", action="store_true", help="Print paper trading summary")
    parser.add_argument("--ev-threshold", type=float, default=0.10)
    parser.add_argument("--stake", type=int, default=1000)
    args = parser.parse_args()

    trader = PaperTrader()

    if args.race_id:
        trader.place_paper_bets(args.race_id, args.ev_threshold, args.stake)
        trader.print_summary()
    elif args.reconcile:
        summary = trader.reconcile()
        trader.print_summary()
    elif args.summary:
        trader.print_summary()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
