"""
UmaEdge — Multi-Day Backtest Dashboard.

Trains the hybrid ensemble once, then evaluates per race day to produce:
  - Per-day P&L breakdown
  - Rolling Sharpe ratio (10-day window)
  - Favorites-only baseline comparison
  - Cumulative equity curve
  - Win streak / consistency metrics

Usage:
    python -m models.multiday_backtest
    python -m models.multiday_backtest --ev-threshold 0.10 --output results/multiday.csv
    python -m models.multiday_backtest --calibration platt --max-odds 20
"""

import argparse
import os
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from scraper.db import get_session
from models.features import FeatureBuilder
from models.backtest import BacktestConfig, Backtester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("multiday_backtest")


# ---------------------------------------------------------------------------
# JRA Constants
# ---------------------------------------------------------------------------

JRA_RACE_DAYS_PER_YEAR = 288  # For annualisation
FEATURES_CACHE = os.path.join(os.path.dirname(__file__), "saved", "features_cache.pkl")


# ---------------------------------------------------------------------------
# Per-Day Result
# ---------------------------------------------------------------------------

@dataclass
class DayResult:
    """Results for a single race day."""
    date: str
    races: int
    bets: int
    wins: int
    staked: int
    payout: int
    profit: int
    roi_pct: float
    fav_bets: int
    fav_wins: int
    fav_profit: int
    fav_roi_pct: float


# ---------------------------------------------------------------------------
# Multi-Day Backtest Engine
# ---------------------------------------------------------------------------

class MultiDayBacktest:
    """Run backtests per race day for granular analysis."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(
        self,
        predictions_df: pd.DataFrame,
    ) -> list[DayResult]:
        """
        Run per-day backtests.

        Args:
            predictions_df: DataFrame with columns:
                race_id, entry_id, win_prob, odds_win, finish_pos, date,
                horse_name (optional), race_name (optional)

        Returns:
            List of DayResult sorted by date
        """
        if "date" not in predictions_df.columns:
            raise ValueError("predictions_df must have a 'date' column")

        bt = Backtester(self.config)
        day_results = []

        for date_str, day_df in sorted(predictions_df.groupby("date")):
            if pd.isna(date_str) or str(date_str) == "None":
                continue

            # --- Model-driven backtest ---
            result = bt.run(day_df, model_version=str(date_str))

            # --- Favorites baseline ---
            fav_bets = 0
            fav_wins = 0
            fav_profit = 0
            flat_stake = self.config.flat_stake or 1000

            for race_id, race_df_inner in day_df.groupby("race_id"):
                valid = race_df_inner[
                    race_df_inner["odds_win"].notna()
                    & (race_df_inner["odds_win"] > 0)
                ]
                if valid.empty:
                    continue
                fav = valid.loc[valid["odds_win"].idxmin()]
                fav_odds = fav["odds_win"]
                fav_finish = fav.get("finish_pos")
                won = fav_finish == 1 if fav_finish is not None else False

                fav_bets += 1
                if won:
                    fav_wins += 1
                    fav_profit += int(flat_stake * fav_odds) - flat_stake
                else:
                    fav_profit -= flat_stake

            day = DayResult(
                date=str(date_str),
                races=len(day_df.groupby("race_id")),
                bets=result.total_bets,
                wins=result.winning_bets,
                staked=result.total_staked,
                payout=result.total_payout,
                profit=result.total_profit,
                roi_pct=result.roi_pct,
                fav_bets=fav_bets,
                fav_wins=fav_wins,
                fav_profit=fav_profit,
                fav_roi_pct=(fav_profit / (fav_bets * flat_stake) * 100)
                if fav_bets > 0 else 0,
            )
            day_results.append(day)

        return sorted(day_results, key=lambda d: d.date)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_rolling_sharpe(
        day_results: list[DayResult], window: int = 10
    ) -> list[tuple[str, float]]:
        """Compute rolling Sharpe ratio over a sliding window of race days."""
        if len(day_results) < window:
            return []

        rois = [d.roi_pct for d in day_results]
        rolling = []
        for i in range(window, len(rois) + 1):
            chunk = np.array(rois[i - window : i])
            if chunk.std() > 0:
                sharpe = chunk.mean() / chunk.std() * np.sqrt(JRA_RACE_DAYS_PER_YEAR)
            else:
                sharpe = 0
            rolling.append((day_results[i - 1].date, round(sharpe, 2)))
        return rolling

    @staticmethod
    def compute_streaks(day_results: list[DayResult]) -> dict:
        """Compute win/loss streak statistics."""
        if not day_results:
            return {}

        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        profitable_days = 0
        losing_days = 0

        for d in day_results:
            if d.profit > 0:
                profitable_days += 1
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            elif d.profit < 0:
                losing_days += 1
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))
            else:
                current_streak = 0

        return {
            "profitable_days": profitable_days,
            "losing_days": losing_days,
            "breakeven_days": len(day_results) - profitable_days - losing_days,
            "win_pct": profitable_days / len(day_results) * 100 if day_results else 0,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(day_results: list[DayResult], rolling_sharpe: list = None):
        """Print a styled multi-day backtest report."""
        if not day_results:
            print("No results to report.")
            return

        print("\n" + "=" * 90)
        print("  UmaEdge — Multi-Day Backtest Dashboard")
        print("=" * 90)

        # Per-day table
        cum_profit = 0
        cum_fav_profit = 0
        print(f"\n  {'Date':<12} {'Races':>5} {'Bets':>4} {'W':>3} "
              f"{'Staked':>8} {'P&L':>8} {'ROI':>7} {'CumP&L':>9} │ "
              f"{'FavW':>4} {'FavP&L':>8} {'FavROI':>7}")
        print("  " + "─" * 86)

        for d in day_results:
            cum_profit += d.profit
            cum_fav_profit += d.fav_profit
            pnl_color = "+" if d.profit >= 0 else ""
            fav_color = "+" if d.fav_profit >= 0 else ""
            cum_color = "+" if cum_profit >= 0 else ""

            print(
                f"  {d.date:<12} {d.races:>5} {d.bets:>4} {d.wins:>3} "
                f"¥{d.staked:>7,} {pnl_color}¥{d.profit:>6,} {d.roi_pct:>+6.1f}% "
                f"{cum_color}¥{cum_profit:>7,} │ "
                f"{d.fav_wins:>4} {fav_color}¥{d.fav_profit:>6,} {d.fav_roi_pct:>+6.1f}%"
            )

        print("  " + "─" * 86)

        # Summary
        total_bets = sum(d.bets for d in day_results)
        total_wins = sum(d.wins for d in day_results)
        total_staked = sum(d.staked for d in day_results)
        total_profit = sum(d.profit for d in day_results)
        total_fav_bets = sum(d.fav_bets for d in day_results)
        total_fav_wins = sum(d.fav_wins for d in day_results)
        total_fav_profit = sum(d.fav_profit for d in day_results)

        overall_roi = total_profit / total_staked * 100 if total_staked else 0
        fav_roi = total_fav_profit / (total_fav_bets * 1000) * 100 if total_fav_bets else 0

        # Streaks
        streaks = MultiDayBacktest.compute_streaks(day_results)

        print(f"\n  {'SUMMARY':<12} {sum(d.races for d in day_results):>5} "
              f"{total_bets:>4} {total_wins:>3} "
              f"¥{total_staked:>7,}  ¥{total_profit:>6,} {overall_roi:>+6.1f}%"
              f"           │ {total_fav_wins:>4}  ¥{total_fav_profit:>6,} {fav_roi:>+6.1f}%")

        print(f"\n  Model vs Favorites:")
        edge = overall_roi - fav_roi
        print(f"    Model ROI:     {overall_roi:>+.1f}%")
        print(f"    Favorites ROI: {fav_roi:>+.1f}%")
        print(f"    Edge:          {edge:>+.1f}pp")

        print(f"\n  Consistency:")
        print(f"    Profitable days: {streaks.get('profitable_days', 0)}/{len(day_results)} "
              f"({streaks.get('win_pct', 0):.0f}%)")
        print(f"    Max win streak:  {streaks.get('max_win_streak', 0)} days")
        print(f"    Max loss streak: {streaks.get('max_loss_streak', 0)} days")

        # Sharpe
        daily_rois = [d.roi_pct for d in day_results]
        if len(daily_rois) >= 2:
            daily_mean = np.mean(daily_rois)
            daily_std = np.std(daily_rois)
            if daily_std > 0:
                annualised_sharpe = daily_mean / daily_std * np.sqrt(JRA_RACE_DAYS_PER_YEAR)
            else:
                annualised_sharpe = 0
            print(f"\n  Sharpe Ratio (annualised): {annualised_sharpe:.2f}")
            print(f"    Daily ROI mean: {daily_mean:+.2f}%  std: {daily_std:.2f}%")

        # Rolling Sharpe
        if rolling_sharpe:
            print(f"\n  Rolling Sharpe (10-day window):")
            print(f"    {'Date':<12} {'Sharpe':>8}")
            for date, sharpe in rolling_sharpe[-10:]:
                print(f"    {date:<12} {sharpe:>8.2f}")

        print("\n" + "=" * 90)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_multiday_backtest(
    ev_threshold: float = 0.10,
    max_odds: float = 30.0,
    min_odds: float = 1.0,
    use_kelly: bool = True,
    calibration_method: str = "isotonic",
    output_path: Optional[str] = None,
    rebuild: bool = False,
):
    """
    Full pipeline:
      1. Build features (or load from cache)
      2. Train hybrid ensemble on pre-2025 data
      3. Generate predictions for 2025+ data
      4. Run per-day backtests
      5. Print dashboard
    """
    from models.ensemble import HybridEnsemble
    import xgboost as xgb

    t0 = time.time()

    # 1. Build features (with parquet caching)
    if not rebuild and os.path.exists(FEATURES_CACHE):
        log.info(f"Step 1: Loading cached features from {FEATURES_CACHE}...")
        features_df = pd.read_pickle(FEATURES_CACHE)
        log.info(f"  Loaded {len(features_df)} entries with {len(features_df.columns)} columns")
    else:
        log.info("Step 1: Building features for all races (this takes ~20 min)...")
        fb = FeatureBuilder()
        features_df = fb.build_features_all()
        if not features_df.empty:
            os.makedirs(os.path.dirname(FEATURES_CACHE), exist_ok=True)
            features_df.to_pickle(FEATURES_CACHE)
            log.info(f"  💾 Cached features to {FEATURES_CACHE}")

    if features_df.empty:
        log.error("No data. Run the scraper first.")
        return

    # Split train/test by year
    features_df["year"] = features_df["date"].str[:4].astype(float)
    train_df = features_df[features_df["year"] < 2025].copy()
    test_df = features_df[features_df["year"] >= 2025].copy()

    log.info(f"Train: {len(train_df)} entries | Test: {len(test_df)} entries")
    log.info(f"Test dates: {test_df['date'].nunique()} unique days")

    if test_df.empty or train_df.empty:
        log.error("Insufficient data for train/test split.")
        return

    # 2. Train hybrid ensemble
    log.info("Step 2: Training hybrid ensemble...")

    ensemble = HybridEnsemble(calibration_method=calibration_method)
    ensemble.train(train_df)

    # 3. Generate predictions
    log.info("Step 3: Generating predictions on test set...")
    preds = ensemble.predict(test_df)
    test_df = test_df.copy()
    test_df["win_prob"] = preds["combined"]

    # Merge odds and results
    with get_session() as session:
        extra = session.execute(text("""
            SELECT
                e.id AS entry_id,
                e.odds_win,
                r.date,
                r.race_name_jp AS race_name,
                h.name_jp AS horse_name,
                res.finish_pos
            FROM horsebet.entries e
            JOIN horsebet.races r ON r.id = e.race_id
            JOIN horsebet.horses h ON h.id = e.horse_id
            LEFT JOIN horsebet.results res ON res.entry_id = e.id
        """)).fetchall()

    extra_df = pd.DataFrame(extra, columns=[
        "entry_id", "odds_win", "date", "race_name", "horse_name", "finish_pos"
    ])

    pred_df = test_df[["race_id", "entry_id", "win_prob"]].merge(
        extra_df, on="entry_id", how="left"
    )

    # 4. Run per-day backtests
    log.info("Step 4: Running per-day backtests...")
    config = BacktestConfig(
        ev_threshold=ev_threshold,
        max_odds=max_odds,
        min_odds=min_odds,
        use_kelly=use_kelly,
        flat_stake=1000,
        # When flat staking, constrain bankroll so Kelly can't inflate bets
        initial_bankroll=1_000_000_000 if use_kelly else 1_000_000,
        max_bet_pct=0.05 if use_kelly else 0.001,  # 0.1% of 1M = ¥1000
    )

    mdb = MultiDayBacktest(config)
    day_results = mdb.run(pred_df)

    # Rolling Sharpe
    rolling_sharpe = mdb.compute_rolling_sharpe(day_results, window=10)

    # 5. Print dashboard
    mdb.print_report(day_results, rolling_sharpe)

    # 6. Export
    if output_path:
        rows = []
        cum_profit = 0
        for d in day_results:
            cum_profit += d.profit
            rows.append({
                "date": d.date,
                "races": d.races,
                "bets": d.bets,
                "wins": d.wins,
                "staked": d.staked,
                "profit": d.profit,
                "roi_pct": round(d.roi_pct, 2),
                "cum_profit": cum_profit,
                "fav_bets": d.fav_bets,
                "fav_wins": d.fav_wins,
                "fav_profit": d.fav_profit,
                "fav_roi_pct": round(d.fav_roi_pct, 2),
            })
        export_df = pd.DataFrame(rows)
        export_df.to_csv(output_path, index=False)
        log.info(f"📄 Exported to {output_path}")

    elapsed = time.time() - t0
    log.info(f"Total time: {elapsed:.0f}s")

    return day_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UmaEdge — Multi-Day Backtest Dashboard"
    )
    parser.add_argument("--ev-threshold", type=float, default=0.10,
                        help="Min EV to trigger bet (default: 0.10)")
    parser.add_argument("--max-odds", type=float, default=30.0,
                        help="Max odds to bet (default: 30.0)")
    parser.add_argument("--min-odds", type=float, default=1.0,
                        help="Min odds to bet (default: 1.0)")
    parser.add_argument("--flat-stake", action="store_true",
                        help="Use flat staking instead of Kelly")
    parser.add_argument("--calibration", type=str, default="isotonic",
                        choices=["none", "platt", "isotonic"],
                        help="Calibration method (default: isotonic)")
    parser.add_argument("--output", type=str,
                        help="Output CSV path for per-day results")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild features (ignore cache)")
    args = parser.parse_args()

    run_multiday_backtest(
        ev_threshold=args.ev_threshold,
        max_odds=args.max_odds,
        min_odds=args.min_odds,
        use_kelly=not args.flat_stake,
        calibration_method=args.calibration,
        output_path=args.output,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
