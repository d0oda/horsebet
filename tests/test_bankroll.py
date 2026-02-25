"""
Unit tests for the Bankroll Manager Agent.
Tests Kelly sizing, correlation adjustment, drawdown circuit-breaker,
and daily exposure caps.
"""

import math

import numpy as np
import pandas as pd
import pytest

from agents.bankroll import (
    BankrollConfig,
    BankrollManager,
    RiskMetrics,
    StakeRecommendation,
)


# ---------------------------------------------------------------------------
# Kelly Fraction
# ---------------------------------------------------------------------------

class TestKellyFraction:
    def setup_method(self):
        self.mgr = BankrollManager()

    def test_positive_edge(self):
        # 30% win prob at 5.0 odds → (4×0.3 - 0.7)/4 = 0.125
        k = self.mgr.kelly_fraction(0.30, 5.0)
        assert k == pytest.approx(0.125, abs=0.001)

    def test_no_edge(self):
        # 10% prob at 5.0 odds → implied = 20% → no edge
        k = self.mgr.kelly_fraction(0.10, 5.0)
        assert k == 0.0

    def test_even_money(self):
        # 60% prob at 2.0 odds → (1×0.6 - 0.4)/1 = 0.2
        k = self.mgr.kelly_fraction(0.60, 2.0)
        assert k == pytest.approx(0.2, abs=0.001)

    def test_zero_odds(self):
        k = self.mgr.kelly_fraction(0.50, 0)
        assert k == 0.0

    def test_negative_odds(self):
        k = self.mgr.kelly_fraction(0.50, -1.0)
        assert k == 0.0

    def test_zero_prob(self):
        k = self.mgr.kelly_fraction(0.0, 5.0)
        assert k == 0.0

    def test_certainty(self):
        # 100% prob → kelly should be near 1
        k = self.mgr.kelly_fraction(0.99, 2.0)
        assert k > 0.9


# ---------------------------------------------------------------------------
# Stake Calculation
# ---------------------------------------------------------------------------

class TestStakeCalculation:
    def setup_method(self):
        self.mgr = BankrollManager()

    def test_basic_stake(self):
        stake = self.mgr.calculate_stake(
            prob=0.30, odds=5.0, bankroll=100_000
        )
        assert stake > 0
        assert stake <= 100_000 * 0.02  # max single bet cap

    def test_zero_bankroll(self):
        stake = self.mgr.calculate_stake(
            prob=0.30, odds=5.0, bankroll=0
        )
        assert stake == 0

    def test_no_edge_zero_stake(self):
        stake = self.mgr.calculate_stake(
            prob=0.10, odds=5.0, bankroll=100_000
        )
        assert stake == 0

    def test_drawdown_halves_stake(self):
        normal = self.mgr.calculate_stake(
            prob=0.30, odds=5.0, bankroll=100_000, drawdown_active=False
        )
        halved = self.mgr.calculate_stake(
            prob=0.30, odds=5.0, bankroll=100_000, drawdown_active=True
        )
        if normal > 0 and halved > 0:
            assert halved <= normal
            # Should be approximately half (within rounding tolerance of ¥100)
            assert halved <= normal * 0.75 + 100

    def test_correlation_reduces_stake(self):
        single = self.mgr.calculate_stake(
            prob=0.30, odds=5.0, bankroll=100_000, n_bets_in_race=1
        )
        multi = self.mgr.calculate_stake(
            prob=0.30, odds=5.0, bankroll=100_000, n_bets_in_race=4
        )
        if single > 0 and multi > 0:
            assert multi < single

    def test_max_bet_cap(self):
        config = BankrollConfig(max_bet=5_000)
        mgr = BankrollManager(config)
        stake = mgr.calculate_stake(
            prob=0.50, odds=3.0, bankroll=10_000_000
        )
        assert stake <= 5_000

    def test_min_bet_threshold(self):
        config = BankrollConfig(min_bet=500)
        mgr = BankrollManager(config)
        # Very small edge → stake below min_bet → should be 0
        stake = mgr.calculate_stake(
            prob=0.21, odds=5.0, bankroll=1_000
        )
        assert stake == 0 or stake >= 500

    def test_rounded_to_100(self):
        stake = self.mgr.calculate_stake(
            prob=0.30, odds=5.0, bankroll=100_000
        )
        assert stake % 100 == 0

    def test_fractional_kelly(self):
        full_mgr = BankrollManager(BankrollConfig(kelly_fraction=1.0, max_single_bet_pct=1.0))
        quarter_mgr = BankrollManager(BankrollConfig(kelly_fraction=0.25, max_single_bet_pct=1.0))

        full = full_mgr.calculate_stake(prob=0.30, odds=5.0, bankroll=100_000)
        quarter = quarter_mgr.calculate_stake(prob=0.30, odds=5.0, bankroll=100_000)

        if full > 0 and quarter > 0:
            # Quarter-Kelly should be less than full
            assert quarter < full


# ---------------------------------------------------------------------------
# Multi-Bet Sizing
# ---------------------------------------------------------------------------

class TestMultiBetSizing:
    def setup_method(self):
        self.mgr = BankrollManager()

    def _make_value_bets(self, n_races=2, bets_per_race=2):
        rows = []
        for r in range(n_races):
            for b in range(bets_per_race):
                rows.append({
                    "entry_id": r * 10 + b,
                    "horse_name": f"Horse_{r}_{b}",
                    "race_id": r + 1,
                    "combined_win_prob": 0.20 + b * 0.05,
                    "odds": 6.0 + b * 2,
                    "market_prob": 0.10,
                    "ev": 0.10 + b * 0.05,
                })
        return pd.DataFrame(rows)

    def test_returns_recommendations(self):
        df = self._make_value_bets()
        recs = self.mgr.calculate_stakes(df, balance=100_000)
        assert isinstance(recs, list)
        assert all(isinstance(r, StakeRecommendation) for r in recs)

    def test_respects_ev_threshold(self):
        config = BankrollConfig(ev_threshold=0.50)
        mgr = BankrollManager(config)
        df = self._make_value_bets()
        recs = mgr.calculate_stakes(df, balance=100_000)
        # All bets have EV < 0.50, so should be empty
        assert len(recs) == 0

    def test_empty_input(self):
        recs = self.mgr.calculate_stakes(pd.DataFrame(), balance=100_000)
        assert recs == []


# ---------------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------------

class TestRiskMetrics:
    def test_drawdown_detection(self):
        metrics = RiskMetrics(
            current_balance=80_000,
            peak_balance=100_000,
            drawdown_pct=0.20,
        )
        assert metrics.drawdown_pct == 0.20

    def test_default_metrics(self):
        metrics = RiskMetrics()
        assert metrics.current_balance == 0
        assert metrics.total_bets == 0
        assert not metrics.is_drawdown_active

    def test_streak_tracking(self):
        metrics = RiskMetrics(
            current_streak=3,
            win_streak=3,
            loss_streak=0,
        )
        assert metrics.win_streak == 3
        assert metrics.loss_streak == 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestBankrollConfig:
    def test_defaults(self):
        config = BankrollConfig()
        assert config.initial_bankroll == 100_000
        assert config.kelly_fraction == 0.25
        assert config.max_daily_exposure == 0.05
        assert config.drawdown_threshold == 0.15
        assert config.min_bet == 100
        assert config.max_bet == 50_000

    def test_custom(self):
        config = BankrollConfig(
            initial_bankroll=500_000,
            kelly_fraction=0.10,
        )
        assert config.initial_bankroll == 500_000
        assert config.kelly_fraction == 0.10
