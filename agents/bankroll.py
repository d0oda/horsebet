"""
UmaEdge — Bankroll Manager Agent.

Automated bankroll sizing engine using fractional Kelly criterion
with correlation adjustments, daily exposure caps, and drawdown
circuit-breakers.

Usage:
    # Show recommended stakes for a race
    python -m agents.bankroll --race-id 1

    # Show P&L summary
    python -m agents.bankroll --summary

    # Record a bet result
    python -m agents.bankroll --record --race-id 1 --entry-id 5 --stake 2000 --payout 8400
"""

import argparse
import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import text

from scraper.db import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bankroll")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BankrollConfig:
    """Parameters for bankroll management."""

    initial_bankroll: int = 100_000       # ¥100,000 starting balance
    kelly_fraction: float = 0.25          # quarter-Kelly for safety
    max_daily_exposure: float = 0.05      # max 5% of bankroll per race day
    max_single_bet_pct: float = 0.02      # max 2% of bankroll on one bet
    drawdown_threshold: float = 0.15      # halve stakes after 15% drawdown
    min_bet: int = 100                    # minimum bet in yen
    max_bet: int = 50_000                 # absolute max per bet
    ev_threshold: float = 0.05            # minimum edge to place a bet


# ---------------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------------

@dataclass
class RiskMetrics:
    """Current risk state of the bankroll."""

    current_balance: int = 0
    peak_balance: int = 0
    drawdown_pct: float = 0.0
    is_drawdown_active: bool = False      # True = stakes halved
    total_bets: int = 0
    winning_bets: int = 0
    total_staked: int = 0
    total_profit: int = 0
    roi_pct: float = 0.0
    win_streak: int = 0
    loss_streak: int = 0
    current_streak: int = 0               # positive = wins, negative = losses
    sharpe_ratio: float = 0.0
    today_exposure: int = 0               # total staked today


# ---------------------------------------------------------------------------
# Stake Recommendation
# ---------------------------------------------------------------------------

@dataclass
class StakeRecommendation:
    """Recommended stake for a single bet."""

    entry_id: int
    horse_name: str
    race_id: int
    model_prob: float
    market_prob: float
    odds: float
    ev: float
    raw_kelly: float                      # full Kelly fraction
    adjusted_kelly: float                 # after fractional + correlation + drawdown
    recommended_stake: int                # final stake in yen
    reason: str = ""                      # explanation for sizing


# ---------------------------------------------------------------------------
# Bankroll Manager
# ---------------------------------------------------------------------------

class BankrollManager:
    """
    Manages bankroll sizing, risk limits, and P&L tracking.

    Implements fractional Kelly criterion with:
    - Correlation adjustment for same-race bets
    - Daily exposure cap
    - Drawdown circuit-breaker
    - Min/max stake enforcement
    """

    def __init__(self, config: BankrollConfig = None):
        self.config = config or BankrollConfig()

    # -------------------------------------------------------------------
    # Kelly Criterion
    # -------------------------------------------------------------------

    def kelly_fraction(self, prob: float, odds: float) -> float:
        """
        Calculate the raw Kelly fraction: f* = (bp - q) / b
        where b = decimal_odds - 1, p = true win probability, q = 1 - p.
        """
        if odds <= 1 or prob <= 0 or prob >= 1:
            return 0.0
        b = odds - 1
        q = 1 - prob
        kelly = (b * prob - q) / b
        return max(0.0, kelly)

    def calculate_stake(
        self,
        prob: float,
        odds: float,
        bankroll: int,
        n_bets_in_race: int = 1,
        drawdown_active: bool = False,
    ) -> int:
        """
        Calculate recommended stake for a single bet.

        Applies:
        1. Raw Kelly → fractional Kelly
        2. Correlation adjustment (1/sqrt(n) for same-race bets)
        3. Drawdown halving
        4. Max single-bet cap
        5. Min/max clamp
        """
        raw_kelly = self.kelly_fraction(prob, odds)
        if raw_kelly <= 0:
            return 0

        # 1. Fractional Kelly
        adj = raw_kelly * self.config.kelly_fraction

        # 2. Correlation adjustment for same-race bets
        if n_bets_in_race > 1:
            adj /= math.sqrt(n_bets_in_race)

        # 3. Drawdown circuit-breaker
        if drawdown_active:
            adj *= 0.5

        # 4. Max single-bet percentage
        adj = min(adj, self.config.max_single_bet_pct)

        # 5. Convert to yen and clamp
        stake = int(bankroll * adj)
        stake = max(stake, 0)

        if stake < self.config.min_bet:
            return 0
        stake = min(stake, self.config.max_bet)

        # Round to nearest 100 yen
        stake = (stake // 100) * 100
        return stake

    # -------------------------------------------------------------------
    # Multi-Bet Sizing
    # -------------------------------------------------------------------

    def calculate_stakes(
        self,
        value_bets: pd.DataFrame,
        balance: Optional[int] = None,
    ) -> list[StakeRecommendation]:
        """
        Calculate recommended stakes for a set of value bets.

        Args:
            value_bets: DataFrame with columns:
                entry_id, horse_name (optional), race_id,
                combined_win_prob (or model_prob), odds, market_prob, ev
            balance: Current bankroll balance. If None, reads from DB.

        Returns:
            List of StakeRecommendation objects
        """
        if value_bets.empty:
            return []

        if balance is None:
            balance = self.get_current_balance()

        risk = self.get_risk_metrics(balance)
        drawdown_active = risk.is_drawdown_active

        # Group by race to count same-race bets
        race_counts = value_bets["race_id"].value_counts().to_dict()
        daily_remaining = int(balance * self.config.max_daily_exposure) - risk.today_exposure

        recommendations = []
        total_staked = 0

        for _, row in value_bets.iterrows():
            race_id = row["race_id"]
            prob = row.get("combined_win_prob", row.get("model_prob", 0))
            odds = row.get("odds", 0)
            market_prob = row.get("market_prob", 0)
            ev = row.get("ev", prob - market_prob)
            entry_id = row.get("entry_id", 0)
            horse_name = str(row.get("horse_name", "Unknown"))

            if ev < self.config.ev_threshold:
                continue

            n_in_race = race_counts.get(race_id, 1)

            stake = self.calculate_stake(
                prob=prob,
                odds=odds,
                bankroll=balance,
                n_bets_in_race=n_in_race,
                drawdown_active=drawdown_active,
            )

            # Enforce daily exposure cap
            if total_staked + stake > daily_remaining:
                stake = max(0, daily_remaining - total_staked)
                stake = (stake // 100) * 100

            if stake <= 0:
                continue

            raw_k = self.kelly_fraction(prob, odds)
            adj_k = stake / balance if balance > 0 else 0

            reason_parts = []
            if drawdown_active:
                reason_parts.append("drawdown-halved")
            if n_in_race > 1:
                reason_parts.append(f"corr-adj({n_in_race} bets in race)")

            recommendations.append(StakeRecommendation(
                entry_id=entry_id,
                horse_name=horse_name,
                race_id=race_id,
                model_prob=prob,
                market_prob=market_prob,
                odds=odds,
                ev=ev,
                raw_kelly=raw_k,
                adjusted_kelly=adj_k,
                recommended_stake=stake,
                reason=", ".join(reason_parts) if reason_parts else "standard",
            ))

            total_staked += stake

        return recommendations

    # -------------------------------------------------------------------
    # Balance & Risk
    # -------------------------------------------------------------------

    def get_current_balance(self) -> int:
        """Read the latest running balance from bankroll_log."""
        try:
            with get_session() as session:
                row = session.execute(text("""
                    SELECT running_balance
                    FROM bankroll_log
                    ORDER BY created_at DESC
                    LIMIT 1
                """)).fetchone()
                if row and row[0] is not None:
                    return int(row[0])
        except Exception as e:
            log.warning(f"Could not read balance from DB: {e}")
        return self.config.initial_bankroll

    def get_risk_metrics(self, current_balance: Optional[int] = None) -> RiskMetrics:
        """Compute current risk metrics from bankroll history."""
        if current_balance is None:
            current_balance = self.get_current_balance()

        metrics = RiskMetrics(current_balance=current_balance)

        try:
            with get_session() as session:
                # Get all bankroll entries
                rows = session.execute(text("""
                    SELECT stake, payout, running_balance, date
                    FROM bankroll_log
                    ORDER BY created_at ASC
                """)).fetchall()

            if not rows:
                metrics.peak_balance = self.config.initial_bankroll
                return metrics

            balances = [r[2] for r in rows if r[2] is not None]
            stakes = [r[0] for r in rows if r[0] is not None]
            payouts = [r[1] or 0 for r in rows]
            profits = [p - s for s, p in zip(stakes, payouts)]

            metrics.total_bets = len(rows)
            metrics.total_staked = sum(stakes)
            metrics.winning_bets = sum(1 for p in payouts if p > 0)
            metrics.total_profit = sum(profits)
            metrics.roi_pct = (
                (metrics.total_profit / metrics.total_staked * 100)
                if metrics.total_staked > 0 else 0
            )

            # Peak balance and drawdown
            peak = self.config.initial_bankroll
            for b in balances:
                peak = max(peak, b)
            metrics.peak_balance = peak
            metrics.drawdown_pct = (
                (peak - current_balance) / peak if peak > 0 else 0
            )
            metrics.is_drawdown_active = metrics.drawdown_pct >= self.config.drawdown_threshold

            # Streak tracking
            streak = 0
            for p in reversed(profits):
                if p > 0:
                    if streak < 0:
                        break
                    streak += 1
                elif p < 0:
                    if streak > 0:
                        break
                    streak -= 1
                else:
                    break
            metrics.current_streak = streak
            metrics.win_streak = max(0, streak)
            metrics.loss_streak = max(0, -streak)

            # Sharpe ratio (daily returns)
            if len(profits) >= 2:
                import numpy as np
                daily_returns = pd.Series(profits)
                mean_r = daily_returns.mean()
                std_r = daily_returns.std()
                metrics.sharpe_ratio = (
                    float(mean_r / std_r * math.sqrt(252))
                    if std_r > 0 else 0
                )

            # Today's exposure
            today = date.today().isoformat()
            metrics.today_exposure = sum(
                r[0] for r in rows
                if r[3] is not None and str(r[3]) == today and r[0] is not None
            )

        except Exception as e:
            log.warning(f"Error computing risk metrics: {e}")
            metrics.peak_balance = self.config.initial_bankroll

        return metrics

    # -------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------

    def record_bet(
        self,
        race_id: int,
        entry_id: int,
        stake: int,
        odds: float,
        bet_type: str = "win",
        combination: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        """Record a placed bet in the bankroll log."""
        balance = self.get_current_balance()
        new_balance = balance - stake

        with get_session() as session:
            session.execute(text("""
                INSERT INTO bankroll_log
                    (date, race_id, bet_type, combination, stake, odds_at_bet,
                     payout, running_balance, notes)
                VALUES
                    (:dt, :race_id, :bet_type, :combo, :stake, :odds,
                     0, :balance, :notes)
            """), {
                "dt": date.today(),
                "race_id": race_id,
                "bet_type": bet_type,
                "combo": combination or str(entry_id),
                "stake": stake,
                "odds": odds,
                "balance": new_balance,
                "notes": notes,
            })

        log.info(f"📝 Recorded bet: ¥{stake:,} on race {race_id} (balance: ¥{new_balance:,})")

    def record_result(
        self,
        race_id: int,
        entry_id: int,
        payout: int,
    ):
        """Update a bet with its result/payout."""
        with get_session() as session:
            # Find the bet
            row = session.execute(text("""
                SELECT id, stake, running_balance
                FROM bankroll_log
                WHERE race_id = :race_id AND combination = :combo
                ORDER BY created_at DESC
                LIMIT 1
            """), {"race_id": race_id, "combo": str(entry_id)}).fetchone()

            if not row:
                log.error(f"No bet found for race {race_id}, entry {entry_id}")
                return

            bet_id, stake, old_balance = row
            new_balance = (old_balance or 0) + payout

            session.execute(text("""
                UPDATE bankroll_log
                SET payout = :payout, running_balance = :balance
                WHERE id = :id
            """), {"payout": payout, "balance": new_balance, "id": bet_id})

        profit = payout - (stake or 0)
        emoji = "🎉" if profit > 0 else "📉"
        log.info(f"{emoji} Result: ¥{payout:,} payout, ¥{profit:+,} profit (balance: ¥{new_balance:,})")

    # -------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------

    def get_daily_summary(self, target_date: Optional[date] = None) -> dict:
        """Get a summary of betting activity for a specific date."""
        target = target_date or date.today()

        with get_session() as session:
            rows = session.execute(text("""
                SELECT stake, payout, bet_type, combination
                FROM bankroll_log
                WHERE date = :dt
                ORDER BY created_at
            """), {"dt": target}).fetchall()

        if not rows:
            return {"date": str(target), "bets": 0, "staked": 0, "payout": 0, "profit": 0}

        total_staked = sum(r[0] for r in rows)
        total_payout = sum(r[1] or 0 for r in rows)

        return {
            "date": str(target),
            "bets": len(rows),
            "staked": total_staked,
            "payout": total_payout,
            "profit": total_payout - total_staked,
            "roi_pct": ((total_payout - total_staked) / total_staked * 100) if total_staked > 0 else 0,
        }

    def print_summary(self, metrics: Optional[RiskMetrics] = None):
        """Print a formatted bankroll summary to stdout."""
        if metrics is None:
            metrics = self.get_risk_metrics()

        print("\n💰 UmaEdge — Bankroll Summary")
        print("=" * 50)
        print(f"  Balance:        ¥{metrics.current_balance:>12,}")
        print(f"  Peak:           ¥{metrics.peak_balance:>12,}")
        print(f"  Drawdown:       {metrics.drawdown_pct:>11.1%}")
        if metrics.is_drawdown_active:
            print(f"  ⚠️  DRAWDOWN ACTIVE — stakes halved")
        print(f"  Total Bets:     {metrics.total_bets:>12,}")
        print(f"  Win Rate:       {(metrics.winning_bets / metrics.total_bets * 100) if metrics.total_bets > 0 else 0:>11.1f}%")
        print(f"  Total Staked:   ¥{metrics.total_staked:>12,}")
        print(f"  Total Profit:   ¥{metrics.total_profit:>+12,}")
        print(f"  ROI:            {metrics.roi_pct:>+11.1f}%")
        print(f"  Sharpe:         {metrics.sharpe_ratio:>12.2f}")
        print(f"  Current Streak: {metrics.current_streak:>+12}")
        print(f"  Today Exposure: ¥{metrics.today_exposure:>12,}")
        print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Bankroll Manager")
    parser.add_argument("--race-id", type=int, help="Race ID to size bets for")
    parser.add_argument("--summary", action="store_true", help="Show bankroll summary")
    parser.add_argument("--record", action="store_true", help="Record a bet/result")
    parser.add_argument("--entry-id", type=int, help="Entry ID (for recording)")
    parser.add_argument("--stake", type=int, help="Stake amount (for recording)")
    parser.add_argument("--payout", type=int, help="Payout amount (for recording result)")
    parser.add_argument("--balance", type=int, help="Override current balance")
    args = parser.parse_args()

    mgr = BankrollManager()

    if args.summary:
        mgr.print_summary()
        daily = mgr.get_daily_summary()
        if daily["bets"] > 0:
            print(f"\n📅 Today: {daily['bets']} bets, ¥{daily['staked']:,} staked, ¥{daily['profit']:+,} P&L")
        return

    if args.record:
        if args.payout is not None and args.race_id and args.entry_id:
            mgr.record_result(args.race_id, args.entry_id, args.payout)
        elif args.stake and args.race_id and args.entry_id:
            mgr.record_bet(args.race_id, args.entry_id, args.stake, odds=0)
        else:
            print("Usage: --record --race-id <id> --entry-id <id> --stake <amount>")
            print("   or: --record --race-id <id> --entry-id <id> --payout <amount>")
        return

    if args.race_id:
        from models.predict import predict_and_store

        balance = args.balance or mgr.get_current_balance()
        predictions = predict_and_store(args.race_id, store_to_db=False)
        value_bets = predictions[predictions["is_value"]]

        if value_bets.empty:
            print(f"No value bets found for race #{args.race_id}")
            return

        recs = mgr.calculate_stakes(value_bets, balance=balance)

        print(f"\n🎯 Stake Recommendations — Race #{args.race_id}")
        print(f"Balance: ¥{balance:,}")
        risk = mgr.get_risk_metrics(balance)
        if risk.is_drawdown_active:
            print("⚠️  DRAWDOWN MODE — stakes halved")
        print("-" * 70)
        print(f"{'Horse':<16} {'Prob':>5} {'Odds':>5} {'EV':>6} {'Kelly':>6} {'Stake':>8} {'Reason'}")
        print("-" * 70)

        total = 0
        for r in recs:
            name = r.horse_name[:14]
            print(
                f"{name:<16} "
                f"{r.model_prob:>4.1%} "
                f"{r.odds:>5.1f} "
                f"{r.ev:>+5.2f} "
                f"{r.adjusted_kelly:>5.2%} "
                f"¥{r.recommended_stake:>7,} "
                f"{r.reason}"
            )
            total += r.recommended_stake

        print("-" * 70)
        print(f"{'Total':<16} {'':<18} {'':>12} ¥{total:>7,}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
