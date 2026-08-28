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
# NOTE (Bug #10): The flat 35% place payout factor was inaccurate across all odds.
# JRA Fukusho payouts scale heavily with win odds: ~12% for 1.5x favorites,
# ~28% for 5x horses, ~50% for 20x+ longshots.
# A static factor is only usable for rough estimates. Use odds-aware function below.


def _jra_place_payout_factor(win_odds: float) -> float:
    """Approximate JRA Fukusho payout as a fraction of win odds.
    Based on empirical JRA pool distributions; range ~0.10 to 0.55.

    Args:
        win_odds: Win market odds (float). If None or <= 0, falls back to the
                  minimum factor (0.12) as a conservative estimate.
                  (Audit R9-MINOR-3): Added None guard to prevent TypeError
                  when odds_win is NULL in the database result.
    """
    if win_odds is None or win_odds <= 0:
        # (Audit R9-MINOR-3): Fallback for NULL/invalid odds — use the minimum
        # payout factor (favorites) as a conservative, safe default.
        return 0.12
    if win_odds <= 2.0:
        return 0.12
    elif win_odds <= 5.0:
        # Linear interpolation: 12% at 2x -> 28% at 5x
        return 0.12 + (win_odds - 2.0) / 3.0 * 0.16
    elif win_odds <= 15.0:
        # 28% at 5x -> 45% at 15x
        return 0.28 + (win_odds - 5.0) / 10.0 * 0.17
    else:
        # 45% at 15x -> 55% cap
        return min(0.55, 0.45 + (win_odds - 15.0) / 30.0 * 0.10)



@dataclass
class BacktestConfig:
    """Parameters for the backtesting simulation."""
    ev_threshold: float = 0.05       # Minimum EV to trigger a bet (5%) — aligned with CLI default
                                     # (Audit INTEGRITY-2: BacktestConfig default was 0.30 (30%) but
                                     # CLI used 0.05, causing incomparable results when called directly.)
    flat_stake: int = 1000           # Yen per flat bet
    initial_bankroll: int = 100000   # Starting bankroll (yen)
    kelly_fraction: float = 0.25    # Quarter-Kelly
    max_bet_pct: float = 0.05       # Max 5% of bankroll per bet
    bet_type: str = "win"           # 'win' or 'place'
    max_odds: float = 30.0          # Skip horses with odds above this
    min_odds: float = 2.0           # Skip horses with odds below this
    use_kelly: bool = True           # Pure Kelly sizing (no flat-stake floor)
    min_kelly_fraction: float = 0.005  # Min Kelly fraction to place a bet
    min_model_prob: float = 0.0     # Minimum model win probability to place a bet


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
    sharpe: float = 0.0          # Per-bet Sharpe (mean/std of per-bet returns)
    daily_sharpe: float = 0.0   # Annualised daily Sharpe (industry standard)
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
        max_dd_pct = 0.0  # track continuous percentage drawdown

        # Sort by date/race
        if "date" in predictions_df.columns:
            predictions_df = predictions_df.sort_values(["date", "race_id", "entry_id"])

        # Group by race — sort=False preserves the chronological sort above.
        # Default sort=True would re-order by race_id integer, causing time-travel.
        races = predictions_df.groupby("race_id", sort=False)
        result.total_races = len(races)

        daily_pnl = {}
        daily_start_balance = {}  # balance at the start of each race day (for daily return denominator)

        for race_id, race_df in races:
            # --- Find the single best bet per race (max 1) ---
            best_candidate = None
            best_ev = -1
            # (Audit R5-INTEGRITY-5): Count actual starters (horses with a recorded finish
            # position) rather than all declared entries.  Scratched horses have
            # finish_pos=None and inflate len(race_df), producing the wrong JRA Fukusho
            # top-N cutoff (top-2 for ≤7 starters vs top-3 for ≥8).
            # (Audit R6-MINOR-1): Use vectorised notna() — O(1) vs O(n) iterrows.
            n_actual_starters = int(race_df["finish_pos"].notna().sum())

            # (Audit R6-MINOR-3): Pre-build the per-race Harville place model ONCE per
            # race rather than rebuilding it inside the per-row loop (O(n²) → O(n)).
            # The JointFinishModel is identical for every entry in the same race.
            _race_place_jfm = None
            _race_place_top_n = 3
            _race_place_probs = {}
            if self.config.bet_type == "place":
                from models.betting_engine import JointFinishModel
                for _, r2 in race_df.iterrows():
                    # (Audit NEW-FLAW-1): post_position absent → fall back to entry_id.
                    _pp = int(r2.get("post_position") or r2.get("entry_id") or 0)
                    _wp = float(r2.get("win_prob", 0) or 0)
                    if _pp > 0 and _wp > 0:
                        _race_place_probs[_pp] = _wp
                if _race_place_probs:
                    _race_place_jfm = JointFinishModel(_race_place_probs)
                    _race_place_top_n = 2 if len(_race_place_probs) <= 7 else 3

            for _, row in race_df.iterrows():
                model_prob = row.get("win_prob", 0)
                odds = row.get("odds_win", 0)

                if not odds or odds <= 0 or (isinstance(odds, float) and np.isnan(odds)):
                    result.nan_odds_skipped += 1
                    continue
                if not model_prob or model_prob <= 0:
                    continue

                # --- Minimum probability filter ---
                if model_prob < self.config.min_model_prob:
                    continue

                # --- Place bet adjustments ---
                # (Audit R4-LOGIC-1): Determine bet type BEFORE the odds filter so that
                # for place bets the floor/ceiling runs on effective place odds, not win
                # odds.  A 2.5x win-odds horse passes min_odds=2.0 but its effective
                # place odds = 2.5 × 0.12 = 0.30x — far below any meaningful floor.
                is_place = self.config.bet_type == "place"
                if is_place:
                    # Use JointFinishModel.place_prob() if a pre-computed race_probs dict
                    # is available on the row; otherwise fall back to the pre-built per-race
                    # Harville model (built once above — R6-MINOR-3).
                    # (Audit FLAW-2: the old model_prob * 2.5 multiplier was arbitrary and
                    # systematically under-estimated place probability for mid-range horses
                    # by up to 12 percentage points versus Harville.)
                    precomputed_place = row.get("place_prob")
                    if precomputed_place is not None and precomputed_place > 0:
                        model_prob = min(float(precomputed_place), 0.99)
                    elif _race_place_jfm is not None:
                        row_pp = int(row.get("post_position") or row.get("entry_id") or 0)
                        if row_pp in _race_place_probs:
                            model_prob = min(_race_place_jfm.place_prob(row_pp, top_n=_race_place_top_n), 0.99)
                        else:
                            # post_position not in race_probs: fall back to 3× win_prob cap
                            model_prob = min(model_prob * 3.0, 0.99)
                    else:
                        model_prob = min(model_prob * 3.0, 0.99)
                    # Fix #10: use odds-aware place payout factor instead of flat 0.35
                    odds = odds * _jra_place_payout_factor(odds)
                    # Now apply floor/ceiling to effective place odds (not raw win odds)
                    if odds > self.config.max_odds or odds < self.config.min_odds:
                        continue
                else:
                    # --- Odds ceiling/floor filter (win bets only) ---
                    if odds > self.config.max_odds or odds < self.config.min_odds:
                        continue

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

                # (Audit R4-INTEGRITY-1): JRA minimum bet unit is ¥100.
                # Kelly can produce sub-¥100 stakes when the bankroll shrinks —
                # e.g. balance=¥20k, kelly_frac=0.006 → stake=¥30.  Recording such
                # bets skews bet-count, hit-rate, and ROI stats.
                if stake < 100:
                    continue
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
                        # Track field size so win condition can mirror the Harville top_n cutoff.
                        # (Audit R3-BONUS): place bet win was hardcoded to finish<=3 even for
                        # <=7-runner fields where JRA only pays Fukusho top-2.
                        "n_race_runners": n_actual_starters,
                    }

            # --- Place the single best bet for this race ---
            if best_candidate is None:
                continue

            c = best_candidate
            row = c["row"]
            finish = row.get("finish_pos")
            if c["is_place"]:
                # JRA Fukusho (place) pays top-2 for fields <=7 runners, top-3 for >=8.
                # (Audit R3-BONUS): was hardcoded to finish<=3, incorrectly crediting
                # 3rd-place finishes in small fields where only top-2 pay.
                n_race_runners = c.get("n_race_runners", len(race_df))
                place_cutoff = 2 if n_race_runners <= 7 else 3
                won = finish is not None and finish <= place_cutoff
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
            # Continuous percentage drawdown (relative to peak at that moment)
            if peak_balance > 0:
                dd_pct = dd / peak_balance
                max_dd_pct = max(max_dd_pct, dd_pct)

            # Daily P&L
            date_str = str(row.get("date", "unknown"))
            if date_str not in daily_start_balance:
                # Record balance at the start of this day (before today's bet)
                daily_start_balance[date_str] = balance - profit
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
        # Use the continuously-tracked percentage (correct) rather than dividing
        # the absolute max drawdown by the final global peak (understates early drawdowns).
        result.max_drawdown_pct = max_dd_pct * 100
        result.avg_ev = np.mean([b.ev for b in result.bets]) if result.bets else 0
        result.avg_odds = np.mean([b.odds for b in result.bets]) if result.bets else 0

        # Daily P&L
        result.daily_pnl = sorted(daily_pnl.items())

        # Per-bet Sharpe — mean(profit/stake) / std(profit/stake)
        # NOTE: structurally depressed by binary payout variance (longshots).
        # Use daily_sharpe for a more meaningful risk-adjusted metric.
        if result.bets:
            valid_bets = [b for b in result.bets if b.stake > 0]
            if len(valid_bets) >= 10:
                bet_returns = np.array([b.profit / b.stake for b in valid_bets])
                if bet_returns.std() > 0:
                    result.sharpe = bet_returns.mean() / bet_returns.std()
                else:
                    result.sharpe = 0
            else:
                result.sharpe = 0
        else:
            result.sharpe = 0  # insufficient data

        # Daily Sharpe — annualised, computed from daily P&L as % of start-of-day balance.
        # Uses the actual balance at the beginning of each race day so that the denominator
        # tracks the compounding bankroll correctly (a true daily return on capital).
        # Previously used a fixed initial_bankroll which understates later returns.
        # Annualisation factor: JRA runs ~104 race days/year (2 per weekend).
        JRA_RACE_DAYS_PER_YEAR = 104

        # Pad daily_pnl with zeros for all unique racing days in the dataset
        # to prevent artificially inflating the mean daily return when bets are rare.
        if predictions_df is not None and "date" in predictions_df.columns:
            all_dates = predictions_df["date"].astype(str).unique()
            for d in all_dates:
                if d not in daily_pnl:
                    daily_pnl[d] = 0.0

        if daily_pnl and len(daily_pnl) >= 5:
            daily_returns = []
            for date_str, pnl in daily_pnl.items():
                # Use the recorded start-of-day balance; fall back to initial_bankroll
                # for zero-bet days (which are padded in) or the very first day.
                denom = daily_start_balance.get(date_str, self.config.initial_bankroll)
                denom = denom if denom > 0 else self.config.initial_bankroll
                daily_returns.append(pnl / denom)
            daily_returns = np.array(daily_returns)
            if daily_returns.std() > 0:
                result.daily_sharpe = (
                    daily_returns.mean() / daily_returns.std()
                ) * np.sqrt(JRA_RACE_DAYS_PER_YEAR)
            else:
                result.daily_sharpe = 0
        else:
            result.daily_sharpe = 0  # insufficient days

        return result

    def _kelly_stake(self, prob: float, odds: float, bankroll: int) -> int:
        """
        Calculate Kelly criterion stake.
        Kelly fraction = (bp - q) / b
        where b = odds - 1, p = win probability, q = 1 - p
        """
        b = odds - 1
        p = prob

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
        print(f"  Sharpe (per-bet):   {result.sharpe:>7.2f}  ← depressed by binary variance")
        print(f"  Sharpe (daily ann): {result.daily_sharpe:>7.2f}  ← annualised, industry standard")
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
    ev_threshold: float = 0.05,  # (Audit R3-INTEGRITY-1): aligned with CLI default; was 0.30, which filtered almost all bets when called from Python directly
    output_path: Optional[str] = None,
    bet_type: str = "win",
    max_odds: float = 30.0,
    min_odds: float = 2.0,
    use_kelly: bool = True,
    use_cache: bool = False,
):
    """
    Run a full backtest using saved model predictions on historical data.
    """
    from models.features import FeatureBuilder
    from models.train import predict_race, load_model
    import os
    import pandas as pd

    if use_cache and os.path.exists("data/features.parquet"):
        log.info("Loading features from data/features.parquet...")
        features_df = pd.read_parquet("data/features.parquet")
    else:
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

    X = features_df[feature_cols].copy()
    
    # Ensure correct types for models
    categorical_features = ["draw", "surface_code", "going_code", "sire_id", "broodmare_sire_id"]
    for col in feature_cols:
        if col in categorical_features or X[col].dtype.name == "category" or X[col].dtype == object:
            X[col] = X[col].fillna("Unknown").astype(str).astype("category")
        elif X[col].dtype == bool:
            X[col] = X[col].astype(int)
            
    # X_vals = X.values if hasattr(X, "values") else X
    lgb_preds = lgb_model.predict(X)
    xgb_preds = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
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
    parser.add_argument("--min-odds", type=float, default=2.0,
                        help="Min odds to bet on (default: 2.0, matches BacktestConfig default)")
    parser.add_argument("--flat-stake", action="store_true",
                        help="Use flat staking instead of Kelly")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use cached features from data/features.parquet")
    args = parser.parse_args()

    run_full_backtest(
        model_version=args.version,
        ev_threshold=args.ev_threshold,
        output_path=args.output,
        bet_type=args.bet_type,
        max_odds=args.max_odds,
        min_odds=args.min_odds,
        use_kelly=not args.flat_stake,
        use_cache=args.use_cache,
    )


if __name__ == "__main__":
    main()
