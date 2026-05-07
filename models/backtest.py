"""
UmaEdge — Historical EV Backtesting Engine.

Simulates a betting strategy over historical data to measure
whether model probabilities translate to positive expected value.

Usage:
    # Run backtest with latest model
    python -m models.backtest

    # Run with specific model version and EV threshold
    python -m models.backtest --version 20250101_120000 --ev-threshold 0.10

    # Output detailed results to CSV
    python -m models.backtest --output results.csv
"""

import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from scraper.db import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backtest")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JRA_TAKE_RATE = 0.25  # JRA takes ~25% of the pool
JRA_PLACE_PAYOUT_FACTOR = 0.35  # Place pays ~35% of win odds on average


@dataclass
class BacktestConfig:
    """Parameters for the backtesting simulation."""
    ev_threshold: float = 0.30       # Minimum EV to trigger a bet (30%)
    flat_stake: int = 1000           # Yen per flat bet
    initial_bankroll: int = 100000   # Starting bankroll (yen)
    kelly_fraction: float = 0.25    # Quarter-Kelly
    max_bet_pct: float = 0.05       # Max 5% of bankroll per bet
    bet_type: str = "win"           # 'win' or 'place'
    max_odds: float = 30.0          # Skip horses with odds above this
    min_odds: float = 2.0           # Skip horses with odds below this
    use_kelly: bool = True           # Pure Kelly sizing (no flat-stake floor)
    min_kelly_fraction: float = 0.005  # Min Kelly fraction to place a bet


@dataclass
class BetRecord:
    """A single bet placed during the backtest."""
    race_id: int
    entry_id: int
    horse_name: str
    date: str
    race_name: str
    model_prob: float
    market_prob: float
    ev: float
    kelly: float
    stake: int
    odds: float
    finish_pos: Optional[int]
    payout: int
    profit: int


@dataclass
class BacktestResult:
    """Summary results of a backtest run."""
    total_races: int = 0
    total_bets: int = 0
    winning_bets: int = 0
    total_staked: int = 0
    total_payout: int = 0
    total_profit: int = 0
    roi_pct: float = 0.0
    hit_rate: float = 0.0
    max_drawdown: int = 0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    avg_ev: float = 0.0
    avg_odds: float = 0.0
    nan_odds_skipped: int = 0
    bets: list = field(default_factory=list)
    daily_pnl: list = field(default_factory=list)
    balance_curve: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """Simulates betting strategies over historical race data."""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        predictions_df: pd.DataFrame,
        model_version: str = "backtest",
    ) -> BacktestResult:
        """
        Run a full backtest.

        Args:
            predictions_df: DataFrame with columns:
                race_id, entry_id, win_prob, odds_win, finish_pos,
                horse_name (optional), date (optional), race_name (optional)
            model_version: Label for this run

        Returns:
            BacktestResult with full stats and bet history
        """
        result = BacktestResult()
        balance = self.config.initial_bankroll
        peak_balance = balance
        max_dd = 0

        # Sort by date/race
        if "date" in predictions_df.columns:
            predictions_df = predictions_df.sort_values(["date", "race_id", "entry_id"])

        # Group by race
        races = predictions_df.groupby("race_id")
        result.total_races = len(races)

        daily_pnl = {}

        for race_id, race_df in races:
            # --- Find the single best bet per race (max 1) ---
            best_candidate = None
            best_ev = -1

            for _, row in race_df.iterrows():
                model_prob = row.get("win_prob", 0)
                odds = row.get("odds_win", 0)

                if not odds or odds <= 0 or (isinstance(odds, float) and np.isnan(odds)):
                    result.nan_odds_skipped += 1
                    continue
                if not model_prob or model_prob <= 0:
                    continue

                # --- Odds ceiling/floor filter ---
                if odds > self.config.max_odds or odds < self.config.min_odds:
                    continue

                # --- Place bet adjustments (Research Backlog #4) ---
                is_place = self.config.bet_type == "place"
                if is_place:
                    model_prob = row.get("place_prob", model_prob * 2.5)
                    model_prob = min(model_prob, 0.99)
                    odds = odds * JRA_PLACE_PAYOUT_FACTOR

                # Implied probability from market odds (after take)
                market_prob = 1.0 / odds

                # EV calculation (ROI)
                ev = (model_prob * odds) - 1.0

                if ev < self.config.ev_threshold:
                    continue

                # Kelly criterion sizing
                kelly = self._kelly_stake(model_prob, odds, balance)

                if self.config.use_kelly:
                    kelly_frac = kelly / balance if balance > 0 else 0
                    if kelly_frac < self.config.min_kelly_fraction:
                        continue
                    stake = min(kelly, int(balance * self.config.max_bet_pct))
                else:
                    stake = self.config.flat_stake

                if stake > balance:
                    continue

                # Track the highest-EV candidate for this race
                if ev > best_ev:
                    best_ev = ev
                    best_candidate = {
                        "row": row,
                        "model_prob": model_prob,
                        "market_prob": market_prob,
                        "ev": ev,
                        "odds": odds,
                        "kelly": kelly,
                        "stake": stake,
                        "is_place": is_place,
                    }

            # --- Place the single best bet for this race ---
            if best_candidate is None:
                continue

            c = best_candidate
            row = c["row"]
            finish = row.get("finish_pos")
            if c["is_place"]:
                won = finish is not None and finish <= 3
            else:
                won = finish == 1 if finish is not None else False
            payout = int(c["stake"] * c["odds"]) if won else 0
            profit = payout - c["stake"]

            bet = BetRecord(
                race_id=race_id,
                entry_id=row.get("entry_id", 0),
                horse_name=row.get("horse_name", "?"),
                date=str(row.get("date", "")),
                race_name=row.get("race_name", ""),
                model_prob=round(c["model_prob"], 4),
                market_prob=round(c["market_prob"], 4),
                ev=round(c["ev"], 4),
                kelly=round(c["kelly"] / balance if balance > 0 else 0, 4),
                stake=c["stake"],
                odds=c["odds"],
                finish_pos=finish,
                payout=payout,
                profit=profit,
            )
            result.bets.append(bet)

            # Update balance
            balance += profit
            result.balance_curve.append(balance)

            # Track drawdown
            peak_balance = max(peak_balance, balance)
            dd = peak_balance - balance
            max_dd = max(max_dd, dd)

            # Daily P&L
            date_str = str(row.get("date", "unknown"))
            daily_pnl[date_str] = daily_pnl.get(date_str, 0) + profit

            if won:
                result.winning_bets += 1

        # Compute summary stats
        result.total_bets = len(result.bets)
        result.total_staked = sum(b.stake for b in result.bets)
        result.total_payout = sum(b.payout for b in result.bets)
        result.total_profit = result.total_payout - result.total_staked
        result.roi_pct = (result.total_profit / result.total_staked * 100) if result.total_staked > 0 else 0
        result.hit_rate = (result.winning_bets / result.total_bets * 100) if result.total_bets > 0 else 0
        result.max_drawdown = max_dd
        result.max_drawdown_pct = (max_dd / peak_balance * 100) if peak_balance > 0 else 0
        result.avg_ev = np.mean([b.ev for b in result.bets]) if result.bets else 0
        result.avg_odds = np.mean([b.odds for b in result.bets]) if result.bets else 0

        # Daily P&L
        result.daily_pnl = sorted(daily_pnl.items())

        # Sharpe ratio — per-bet percentage returns (profit / stake)
        # Un-annualized: mean(return) / std(return) across individual bets
        # Requires minimum 10 bets for statistical meaning
        if result.bets and len(result.bets) >= 10:
            bet_returns = np.array([b.profit / b.stake for b in result.bets])
            if bet_returns.std() > 0:
                result.sharpe = bet_returns.mean() / bet_returns.std()
            else:
                result.sharpe = 0
        else:
            result.sharpe = 0  # insufficient data

        return result

    def _kelly_stake(self, prob: float, odds: float, bankroll: int) -> int:
        """
        Calculate Kelly criterion stake.
        Kelly fraction = (bp - q) / b
        where b = odds - 1, p = win probability, q = 1 - p
        """
        b = odds - 1
        p = prob
        q = 1 - p

        if b <= 0:
            return 0

        ev = (p * odds) - 1.0
        kelly = ev / b
        kelly = max(0, kelly)  # don't bet negative edge
        kelly *= self.config.kelly_fraction  # fractional Kelly

        return int(bankroll * kelly)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(result: BacktestResult):
        """Print a formatted backtest report."""
        print("\n" + "=" * 60)
        print("  UmaEdge — Backtest Report")
        print("=" * 60)
        print(f"  Races analysed:     {result.total_races:>8}")
        print(f"  Bets placed:        {result.total_bets:>8}")
        print(f"  NaN-odds skipped:   {result.nan_odds_skipped:>8}")
        print(f"  Hit rate:           {result.hit_rate:>7.1f}%")
        print(f"  Avg EV per bet:     {result.avg_ev:>7.3f}")
        print(f"  Avg odds:           {result.avg_odds:>7.1f}x")
        print("-" * 60)
        print(f"  Total staked:       ¥{result.total_staked:>10,}")
        print(f"  Total payout:       ¥{result.total_payout:>10,}")
        print(f"  Total profit:       ¥{result.total_profit:>10,}")
        print(f"  ROI:                {result.roi_pct:>7.1f}%")
        print("-" * 60)
        print(f"  Max drawdown:       ¥{result.max_drawdown:>10,} ({result.max_drawdown_pct:.1f}%)")
        print(f"  Sharpe ratio:       {result.sharpe:>7.2f}")
        print("=" * 60)

        if result.bets:
            print(f"\n  Last 10 bets:")
            print(f"  {'Date':<12} {'Horse':<14} {'P(win)':>6} {'Odds':>5} {'EV':>6} {'Stake':>6} {'P&L':>7}")
            print("  " + "-" * 58)
            for b in result.bets[-10:]:
                pnl_str = f"+{b.profit}" if b.profit >= 0 else str(b.profit)
                win_marker = "✅" if b.profit > 0 else "❌"
                print(
                    f"  {b.date:<12} {b.horse_name:<14} "
                    f"{b.model_prob:>5.1%} {b.odds:>5.1f} {b.ev:>+5.2f} "
                    f"¥{b.stake:>5,} {pnl_str:>7} {win_marker}"
                )

    @staticmethod
    def to_dataframe(result: BacktestResult) -> pd.DataFrame:
        """Convert bet history to a DataFrame."""
        if not result.bets:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "date": b.date,
                "race_id": b.race_id,
                "horse": b.horse_name,
                "model_prob": b.model_prob,
                "market_prob": b.market_prob,
                "ev": b.ev,
                "odds": b.odds,
                "stake": b.stake,
                "finish": b.finish_pos,
                "payout": b.payout,
                "profit": b.profit,
            }
            for b in result.bets
        ])


# ---------------------------------------------------------------------------
# Standalone Backtest with Model Predictions
# ---------------------------------------------------------------------------

def run_full_backtest(
    model_version: str = "latest",
    ev_threshold: float = 0.30,
    output_path: Optional[str] = None,
    bet_type: str = "win",
    max_odds: float = 30.0,
    min_odds: float = 2.0,
    use_kelly: bool = True,
):
    """
    Run a full backtest using saved model predictions on historical data.
    """
    from models.features import FeatureBuilder
    from models.train import predict_race, load_model

    log.info("Building features for all historical races...")
    fb = FeatureBuilder()
    features_df = fb.build_features_all()

    if features_df.empty:
        log.error("No data to backtest. Run the scraper first.")
        return

    log.info("Loading model and generating predictions...")
    lgb_model, xgb_model, meta = load_model(model_version)
    feature_cols = meta["feature_cols"]

    import xgboost as xgb

    X = features_df[feature_cols].fillna(0).values
    lgb_preds = lgb_model.predict(X)
    xgb_preds = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_cols))
    features_df["win_prob"] = 0.55 * lgb_preds + 0.45 * xgb_preds

    # Merge with odds and results
    pred_df = features_df[["race_id", "entry_id", "win_prob"]].copy()

    # Load odds and results from DB
    with get_session() as session:
        data = session.execute(text("""
            SELECT
                e.id AS entry_id,
                e.odds_win,
                r.date,
                r.race_name_jp AS race_name,
                h.name_jp AS horse_name,
                res.finish_pos
            FROM entries e
            JOIN races r ON r.id = e.race_id
            JOIN horses h ON h.id = e.horse_id
            LEFT JOIN results res ON res.entry_id = e.id
        """)).fetchall()

    extra_df = pd.DataFrame(data, columns=["entry_id", "odds_win", "date", "race_name", "horse_name", "finish_pos"])
    pred_df = pred_df.merge(extra_df, on="entry_id", how="left")

    # Run backtest
    config = BacktestConfig(
        ev_threshold=ev_threshold,
        bet_type=bet_type,
        max_odds=max_odds,
        min_odds=min_odds,
        use_kelly=use_kelly,
    )
    bt = Backtester(config)
    result = bt.run(pred_df)

    bt.print_report(result)

    if output_path:
        df = bt.to_dataframe(result)
        df.to_csv(output_path, index=False)
        log.info(f"📄 Results saved to {output_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Backtesting Engine")
    parser.add_argument("--version", type=str, default="latest", help="Model version to use")
    parser.add_argument("--ev-threshold", type=float, default=0.05, help="Min EV to trigger bet")
    parser.add_argument("--output", type=str, help="Output CSV path")
    parser.add_argument("--bet-type", type=str, default="win", choices=["win", "place"],
                        help="Bet type: 'win' or 'place' (default: win)")
    parser.add_argument("--max-odds", type=float, default=30.0,
                        help="Max odds to bet on (default: 30.0)")
    parser.add_argument("--min-odds", type=float, default=1.0,
                        help="Min odds to bet on (default: 1.0)")
    parser.add_argument("--flat-stake", action="store_true",
                        help="Use flat staking instead of Kelly")
    args = parser.parse_args()

    run_full_backtest(
        model_version=args.version,
        ev_threshold=args.ev_threshold,
        output_path=args.output,
        bet_type=args.bet_type,
        max_odds=args.max_odds,
        min_odds=args.min_odds,
        use_kelly=not args.flat_stake,
    )


if __name__ == "__main__":
    main()
