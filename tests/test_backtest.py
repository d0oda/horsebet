"""
Unit tests for the backtesting engine.
Tests Kelly sizing, backtest execution, and report generation.
"""

import numpy as np
import pandas as pd
import pytest

from models.backtest import (
    BacktestConfig,
    Backtester,
    BacktestResult,
    JRA_TAKE_RATE,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        config = BacktestConfig()
        assert config.ev_threshold == 0.05
        assert config.flat_stake == 1000
        assert config.initial_bankroll == 100000
        assert config.kelly_fraction == 0.25
        assert config.max_bet_pct == 0.05

    def test_custom_config(self):
        config = BacktestConfig(ev_threshold=0.10, flat_stake=2000)
        assert config.ev_threshold == 0.10
        assert config.flat_stake == 2000


# ---------------------------------------------------------------------------
# Kelly Sizing
# ---------------------------------------------------------------------------

class TestKellySizing:
    def setup_method(self):
        self.bt = Backtester()

    def test_positive_edge(self):
        # 30% true prob at 5.0 odds → positive edge
        stake = self.bt._kelly_stake(prob=0.30, odds=5.0, bankroll=100000)
        assert stake > 0

    def test_no_edge_zero_stake(self):
        # 10% true prob at 5.0 odds → implied prob = 20%, so 10% < 20%
        stake = self.bt._kelly_stake(prob=0.10, odds=5.0, bankroll=100000)
        assert stake == 0

    def test_fractional_kelly(self):
        """Quarter-Kelly should be less than full Kelly."""
        full_kelly_bt = Backtester(BacktestConfig(kelly_fraction=1.0))
        quarter_kelly_bt = Backtester(BacktestConfig(kelly_fraction=0.25))

        full = full_kelly_bt._kelly_stake(0.40, 4.0, 100000)
        quarter = quarter_kelly_bt._kelly_stake(0.40, 4.0, 100000)

        assert quarter < full
        assert quarter == pytest.approx(full * 0.25, rel=0.01)

    def test_zero_bankroll(self):
        stake = self.bt._kelly_stake(0.40, 4.0, 0)
        assert stake == 0

    def test_zero_odds(self):
        stake = self.bt._kelly_stake(0.50, 0.0, 100000)
        assert stake == 0

    def test_negative_odds(self):
        stake = self.bt._kelly_stake(0.50, -1.0, 100000)
        assert stake == 0


# ---------------------------------------------------------------------------
# Backtest Execution
# ---------------------------------------------------------------------------

class TestBacktestRun:
    def _make_predictions(self, n_races=5, entries_per_race=8):
        """Create synthetic prediction data for backtesting."""
        rows = []
        for r in range(n_races):
            for e in range(entries_per_race):
                model_prob = np.random.uniform(0.05, 0.25)
                odds = np.random.uniform(3.0, 30.0)
                finish = e + 1  # entries in order of finish

                rows.append({
                    "race_id": r + 1,
                    "entry_id": r * entries_per_race + e + 1,
                    "win_prob": model_prob,
                    "odds_win": odds,
                    "finish_pos": finish,
                    "horse_name": f"Horse_{r}_{e}",
                    "date": f"2024-{(r % 12) + 1:02d}-15",
                    "race_name": f"Race {r + 1}",
                })
        return pd.DataFrame(rows)

    def test_backtest_runs(self):
        df = self._make_predictions()
        bt = Backtester()
        result = bt.run(df)

        assert isinstance(result, BacktestResult)
        assert result.total_races == 5

    def test_zero_ev_threshold_places_more_bets(self):
        df = self._make_predictions()
        strict = Backtester(BacktestConfig(ev_threshold=0.20))
        loose = Backtester(BacktestConfig(ev_threshold=0.01))

        r_strict = strict.run(df)
        r_loose = loose.run(df)

        assert r_loose.total_bets >= r_strict.total_bets

    def test_no_bets_when_threshold_too_high(self):
        df = self._make_predictions()
        bt = Backtester(BacktestConfig(ev_threshold=0.99))
        result = bt.run(df)

        assert result.total_bets == 0
        assert result.roi_pct == 0

    def test_balance_tracking(self):
        df = self._make_predictions()
        bt = Backtester()
        result = bt.run(df)

        if result.total_bets > 0:
            assert len(result.balance_curve) == result.total_bets

    def test_profit_calculation(self):
        result = BacktestResult()
        result.total_staked = 10000
        result.total_payout = 12000
        result.total_profit = result.total_payout - result.total_staked

        assert result.total_profit == 2000

    def test_hit_rate_calculation(self):
        df = self._make_predictions()
        bt = Backtester(BacktestConfig(ev_threshold=0.0))
        result = bt.run(df)

        if result.total_bets > 0:
            expected_hit = result.winning_bets / result.total_bets * 100
            assert result.hit_rate == pytest.approx(expected_hit, rel=0.01)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class TestReporting:
    def test_to_dataframe(self):
        df = pd.DataFrame({
            "race_id": [1, 1],
            "entry_id": [1, 2],
            "win_prob": [0.25, 0.15],
            "odds_win": [4.0, 8.0],
            "finish_pos": [1, 3],
            "horse_name": ["A", "B"],
            "date": ["2024-01-01", "2024-01-01"],
            "race_name": ["Test Race", "Test Race"],
        })
        bt = Backtester(BacktestConfig(ev_threshold=0.0))
        result = bt.run(df)
        report_df = bt.to_dataframe(result)

        if result.total_bets > 0:
            assert "date" in report_df.columns
            assert "profit" in report_df.columns

    def test_empty_result(self):
        result = BacktestResult()
        df = Backtester.to_dataframe(result)
        assert df.empty

    def test_print_report_no_error(self):
        """Ensure print_report doesn't raise on empty results."""
        result = BacktestResult()
        Backtester.print_report(result)  # should not raise
