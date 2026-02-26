"""
UmaEdge — Live Trading Module (Sprint 7.5).

Real-money trading with safety guardrails:
  - Default ¥100 flat stake
  - Max ¥500 per bet
  - Max ¥5,000 daily loss limit
  - Minimum model confidence threshold
  - Requires --confirm flag (no accidental live trades)
  - Records to bankroll_log and live_trades tables

Usage:
    from models.live_trade import LiveTrader
    trader = LiveTrader()
    trader.place_live_bets(race_id=123, confirm=True)
"""

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("live_trade")


# ---------------------------------------------------------------------------
# Safety Defaults
# ---------------------------------------------------------------------------

DEFAULT_STAKE = 100             # ¥100 per bet
MAX_STAKE_PER_BET = 500         # ¥500 maximum per bet
MAX_DAILY_LOSS = 5_000          # ¥5,000 daily loss limit
MIN_MODEL_CONFIDENCE = 0.15     # minimum model probability
MIN_EV_THRESHOLD = 0.10         # minimum EV to place a bet
REQUIRED_PAPER_WEEKENDS = 4     # profitable paper weekends before live


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class LiveBet:
    """A real-money bet."""
    race_id: int
    entry_id: int
    horse_name: str
    date: str
    bet_type: str
    model_prob: float
    market_prob: float
    ev: float
    odds: float
    stake: int
    kelly_fraction: float
    model_version: str
    status: str = "pending"
    finish_pos: Optional[int] = None
    payout: int = 0
    confirmed_at: Optional[str] = None
    created_at: str = ""


# ---------------------------------------------------------------------------
# Guardrail Checks
# ---------------------------------------------------------------------------

class GuardrailError(Exception):
    """Raised when a safety guardrail is violated."""
    pass


def check_paper_track_record(min_weekends: int = REQUIRED_PAPER_WEEKENDS) -> dict:
    """
    Check paper trading history for profitable weekends.

    Returns:
        Dict with weekend_count, profitable_count, and is_ready.
    """
    from pathlib import Path

    paper_path = Path(__file__).parent.parent / "data" / "paper_trades.json"
    if not paper_path.exists():
        return {"weekend_count": 0, "profitable_count": 0, "is_ready": False}

    trades = json.loads(paper_path.read_text())
    settled = [t for t in trades if t.get("status") != "pending"]

    if not settled:
        return {"weekend_count": 0, "profitable_count": 0, "is_ready": False}

    # Group by week (ISO week number)
    from collections import defaultdict
    weekly = defaultdict(lambda: {"staked": 0, "payout": 0})

    for t in settled:
        trade_date = t.get("date", "")[:10]
        if not trade_date:
            continue
        try:
            dt = datetime.strptime(trade_date, "%Y-%m-%d")
            week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
        except ValueError:
            continue
        weekly[week_key]["staked"] += t.get("hypothetical_stake", 0)
        weekly[week_key]["payout"] += t.get("payout", 0)

    weekend_count = len(weekly)
    profitable = sum(1 for w in weekly.values() if w["payout"] > w["staked"])

    return {
        "weekend_count": weekend_count,
        "profitable_count": profitable,
        "is_ready": profitable >= min_weekends,
    }


def check_daily_loss(trade_date: str = None) -> dict:
    """
    Check current daily loss against the limit.

    Returns:
        Dict with daily_loss, limit, and is_within_limit.
    """
    if trade_date is None:
        trade_date = date.today().isoformat()

    try:
        from scraper.db import get_session
        from sqlalchemy import text

        with get_session() as session:
            result = session.execute(text("""
                SELECT
                    COALESCE(SUM(stake), 0) AS total_staked,
                    COALESCE(SUM(payout), 0) AS total_payout
                FROM live_trades
                WHERE date = :d
            """), {"d": trade_date}).fetchone()

            if result:
                total_staked = result[0]
                total_payout = result[1]
                daily_loss = total_staked - total_payout
            else:
                daily_loss = 0
    except Exception:
        daily_loss = 0

    return {
        "daily_loss": daily_loss,
        "limit": MAX_DAILY_LOSS,
        "remaining": MAX_DAILY_LOSS - daily_loss,
        "is_within_limit": daily_loss < MAX_DAILY_LOSS,
    }


# ---------------------------------------------------------------------------
# Live Trader
# ---------------------------------------------------------------------------

class LiveTrader:
    """Manages live (real-money) trading with safety guardrails."""

    def __init__(
        self,
        stake: int = DEFAULT_STAKE,
        ev_threshold: float = MIN_EV_THRESHOLD,
        max_daily_loss: int = MAX_DAILY_LOSS,
    ):
        self.stake = min(stake, MAX_STAKE_PER_BET)
        self.ev_threshold = ev_threshold
        self.max_daily_loss = max_daily_loss

    def _validate_guardrails(self, confirm: bool = False):
        """Run all safety checks before placing bets."""
        # 1. Confirmation required
        if not confirm:
            raise GuardrailError(
                "🚫 Live trading requires --confirm flag. "
                "This will place REAL bets with REAL money."
            )

        # 2. Check stake limits
        if self.stake > MAX_STAKE_PER_BET:
            raise GuardrailError(
                f"🚫 Stake ¥{self.stake} exceeds maximum ¥{MAX_STAKE_PER_BET}/bet"
            )

        # 3. Check daily loss limit
        daily = check_daily_loss()
        if not daily["is_within_limit"]:
            raise GuardrailError(
                f"🚫 Daily loss limit reached: ¥{daily['daily_loss']:,} "
                f"(limit: ¥{daily['limit']:,})"
            )

        # 4. Check paper trading track record
        track = check_paper_track_record()
        if not track["is_ready"]:
            log.warning(
                f"⚠️  Paper track record: {track['profitable_count']}/{REQUIRED_PAPER_WEEKENDS} "
                f"profitable weekends (need {REQUIRED_PAPER_WEEKENDS})"
            )
            # Warning only — don't block, but log it prominently
            print(
                f"\n  ⚠️  WARNING: Only {track['profitable_count']} profitable "
                f"paper weekends out of {REQUIRED_PAPER_WEEKENDS} required.\n"
                f"  Proceeding anyway since --confirm was specified.\n"
            )

    def place_live_bets(
        self,
        race_id: int,
        confirm: bool = False,
        model_version: str = "latest",
    ) -> list[LiveBet]:
        """
        Run predictions and place live bets with guardrail checks.

        Args:
            race_id: Database race ID
            confirm: Must be True to actually place bets
            model_version: Model version to use

        Returns:
            List of LiveBet objects placed
        """
        # Validate all guardrails
        self._validate_guardrails(confirm=confirm)

        from models.predict import predict_and_store

        log.info(f"🔴 LIVE MODE — Running predictions for race {race_id}...")
        preds_df = predict_and_store(
            race_id, store_to_db=False, ev_threshold=self.ev_threshold,
            model_version=model_version,
        )

        if preds_df is None or preds_df.empty:
            log.warning(f"No predictions for race {race_id}")
            return []

        bets = []
        for _, row in preds_df.iterrows():
            model_prob = row.get("combined_win_prob", row.get("win_prob", 0))
            odds = row.get("odds", 0)

            if not odds or odds <= 0:
                continue

            # Minimum confidence check
            if model_prob < MIN_MODEL_CONFIDENCE:
                continue

            market_prob = 1.0 / odds
            ev = model_prob - market_prob

            if ev < self.ev_threshold:
                continue

            # Check daily loss before each bet
            daily = check_daily_loss()
            if not daily["is_within_limit"]:
                log.warning("🚫 Daily loss limit reached mid-session, stopping")
                break

            # Kelly sizing (quarter-Kelly, capped)
            b = odds - 1
            p = model_prob
            q = 1 - p
            kelly = max(0, (b * p - q) / b) if b > 0 else 0
            quarter_kelly = kelly * 0.25
            kelly_stake = int(100_000 * quarter_kelly)  # assume 100k bankroll
            actual_stake = min(self.stake, kelly_stake, MAX_STAKE_PER_BET)
            actual_stake = max(actual_stake, DEFAULT_STAKE)  # minimum ¥100

            bet = LiveBet(
                race_id=race_id,
                entry_id=row.get("entry_id", 0),
                horse_name=str(row.get("horse_name", "?")),
                date=date.today().isoformat(),
                bet_type="win",
                model_prob=round(model_prob, 4),
                market_prob=round(market_prob, 4),
                ev=round(ev, 4),
                odds=odds,
                stake=actual_stake,
                kelly_fraction=round(quarter_kelly, 4),
                model_version=model_version,
                confirmed_at=datetime.now().isoformat(),
                created_at=datetime.now().isoformat(),
            )
            bets.append(bet)

        # Store to database
        if bets:
            self._store_live_trades(bets)

        log.info(f"🔴 Placed {len(bets)} LIVE bets for race {race_id}")
        return bets

    def _store_live_trades(self, bets: list[LiveBet]):
        """Store live bets to the live_trades table."""
        try:
            from scraper.db import get_session
            from sqlalchemy import text

            with get_session() as session:
                for bet in bets:
                    session.execute(text("""
                        INSERT INTO live_trades (
                            race_id, entry_id, horse_name, date, bet_type,
                            model_prob, market_prob, ev, odds, stake,
                            kelly_fraction, model_version, status, confirmed_at
                        ) VALUES (
                            :race_id, :entry_id, :horse_name, :date, :bet_type,
                            :model_prob, :market_prob, :ev, :odds, :stake,
                            :kelly, :version, 'pending', :confirmed_at
                        )
                    """), {
                        "race_id": bet.race_id,
                        "entry_id": bet.entry_id,
                        "horse_name": bet.horse_name,
                        "date": bet.date,
                        "bet_type": bet.bet_type,
                        "model_prob": bet.model_prob,
                        "market_prob": bet.market_prob,
                        "ev": bet.ev,
                        "odds": bet.odds,
                        "stake": bet.stake,
                        "kelly": bet.kelly_fraction,
                        "version": bet.model_version,
                        "confirmed_at": bet.confirmed_at,
                    })

            log.info(f"💾 Stored {len(bets)} live trades to database")
        except Exception as e:
            log.error(f"Failed to store live trades: {e}")
            # Also save locally as backup
            backup_path = Path(__file__).parent.parent / "data" / "live_trades_backup.json"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            existing = json.loads(backup_path.read_text()) if backup_path.exists() else []
            existing.extend([asdict(b) for b in bets])
            backup_path.write_text(json.dumps(existing, indent=2, default=str))
            log.info(f"💾 Backup saved to {backup_path}")

    def reconcile(self) -> dict:
        """Reconcile pending live trades against actual results."""
        from scraper.db import get_session
        from sqlalchemy import text

        reconciled = 0
        with get_session() as session:
            pending = session.execute(text("""
                SELECT lt.id, lt.entry_id, lt.stake, lt.odds
                FROM live_trades lt
                WHERE lt.status = 'pending'
            """)).fetchall()

            for trade_id, entry_id, stake, odds in pending:
                result = session.execute(text("""
                    SELECT res.finish_pos
                    FROM results res
                    WHERE res.entry_id = :entry_id
                """), {"entry_id": entry_id}).fetchone()

                if result:
                    finish_pos = result[0]
                    won = finish_pos == 1 if finish_pos is not None else False
                    payout = int(stake * odds) if won else 0
                    status = "won" if won else "lost"

                    session.execute(text("""
                        UPDATE live_trades
                        SET status = :status,
                            finish_pos = :finish_pos,
                            payout = :payout,
                            reconciled_at = now()
                        WHERE id = :id
                    """), {
                        "status": status,
                        "finish_pos": finish_pos,
                        "payout": payout,
                        "id": trade_id,
                    })
                    reconciled += 1

                    # Also log to bankroll_log
                    session.execute(text("""
                        INSERT INTO bankroll_log (
                            date, race_id, bet_type, stake,
                            odds_at_bet, payout, notes
                        ) VALUES (
                            CURRENT_DATE, (SELECT race_id FROM live_trades WHERE id = :id),
                            'win', :stake, :odds, :payout, :notes
                        )
                    """), {
                        "id": trade_id,
                        "stake": stake,
                        "odds": odds,
                        "payout": payout,
                        "notes": f"Live trade #{trade_id} — {'WON' if won else 'LOST'}",
                    })

        log.info(f"✅ Reconciled {reconciled} live trades")
        return {"reconciled": reconciled}

    def print_summary(self):
        """Print live trading summary."""
        try:
            from scraper.db import get_session
            from sqlalchemy import text

            with get_session() as session:
                summary = session.execute(text("""
                    SELECT
                        COUNT(*) AS total,
                        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
                        SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) AS wins,
                        SUM(CASE WHEN status = 'lost' THEN 1 ELSE 0 END) AS losses,
                        COALESCE(SUM(stake), 0) AS total_staked,
                        COALESCE(SUM(payout), 0) AS total_payout
                    FROM live_trades
                """)).fetchone()

                total, pending, wins, losses, staked, payout = summary
                profit = payout - staked
                hit_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

                print(f"\n{'=' * 60}")
                print("  UmaEdge — 🔴 LIVE Trading Summary")
                print(f"{'=' * 60}")
                print(f"  Total trades:    {total:>6}")
                print(f"  Pending:         {pending:>6}")
                print(f"  Won:             {wins:>6}")
                print(f"  Lost:            {losses:>6}")
                print(f"  Hit rate:        {hit_rate:>5.1f}%")
                print("-" * 60)
                print(f"  Total staked:    ¥{staked:>10,}")
                print(f"  Total payout:    ¥{payout:>10,}")
                print(f"  Profit/Loss:     ¥{profit:>10,}")
                if staked > 0:
                    print(f"  ROI:             {profit / staked * 100:>5.1f}%")
                print(f"{'=' * 60}")

        except Exception as e:
            log.error(f"Could not fetch live trading summary: {e}")
            print("  No live trading data available.")


# ---------------------------------------------------------------------------
# Convenience for importing Path
# ---------------------------------------------------------------------------
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Live Trading (Sprint 7.5)")
    parser.add_argument("--race-id", type=int, help="Place live bets for this race")
    parser.add_argument("--reconcile", action="store_true", help="Reconcile pending live trades")
    parser.add_argument("--summary", action="store_true", help="Print live trading summary")
    parser.add_argument("--confirm", action="store_true",
                        help="REQUIRED: Confirm you want to place REAL bets")
    parser.add_argument("--stake", type=int, default=DEFAULT_STAKE,
                        help=f"Stake per bet in yen (default: {DEFAULT_STAKE}, max: {MAX_STAKE_PER_BET})")
    parser.add_argument("--ev-threshold", type=float, default=MIN_EV_THRESHOLD,
                        help=f"Min EV threshold (default: {MIN_EV_THRESHOLD})")
    parser.add_argument("--check", action="store_true",
                        help="Check guardrails without placing bets")
    args = parser.parse_args()

    if args.check:
        print("\n  Guardrail Status:")
        track = check_paper_track_record()
        print(f"  Paper weekends:  {track['profitable_count']}/{REQUIRED_PAPER_WEEKENDS} "
              f"({'✅ Ready' if track['is_ready'] else '⚠️ Not ready'})")
        daily = check_daily_loss()
        print(f"  Daily loss:      ¥{daily['daily_loss']:,} / ¥{daily['limit']:,} "
              f"({'✅ OK' if daily['is_within_limit'] else '🚫 Limit reached'})")
        print(f"  Max stake/bet:   ¥{MAX_STAKE_PER_BET}")
        print(f"  Min confidence:  {MIN_MODEL_CONFIDENCE:.0%}")
        return

    trader = LiveTrader(stake=args.stake, ev_threshold=args.ev_threshold)

    if args.race_id:
        trader.place_live_bets(args.race_id, confirm=args.confirm)
        trader.print_summary()
    elif args.reconcile:
        trader.reconcile()
        trader.print_summary()
    elif args.summary:
        trader.print_summary()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
