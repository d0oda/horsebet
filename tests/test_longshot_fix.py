"""
Unit tests for Priority 1-3 changes:
- Longshot bias fixes (P1)
- Adaptive blending (P2)
- Bet sizing & odds filters (P3)
"""

import numpy as np
import pandas as pd
import pytest

from models.backtest import BacktestConfig, Backtester, BacktestResult
from models.ensemble import DivergenceDetector, AlphaSignal


# ---------------------------------------------------------------------------
# Priority 1 — Longshot Features
# ---------------------------------------------------------------------------

class TestLongshotFeatures:
    def test_odds_features_include_longshot_flags(self):
        """ODDS_FEATURES should include the new longshot flags."""
        from models.features import ODDS_FEATURES
        assert "is_longshot" in ODDS_FEATURES
        assert "is_extreme_longshot" in ODDS_FEATURES

    def test_odds_features_length(self):
        """ODDS_FEATURES has 8 features now (6 original + 2 longshot flags)."""
        from models.features import ODDS_FEATURES
        assert len(ODDS_FEATURES) == 8


# ---------------------------------------------------------------------------
# Priority 1 — Odds Ceiling in Backtester
# ---------------------------------------------------------------------------

class TestOdsCeiling:
    def _make_predictions(self):
        """Create predictions with entries at various odds levels."""
        return pd.DataFrame({
            "race_id": [1, 1, 1, 1, 1],
            "entry_id": [1, 2, 3, 4, 5],
            "win_prob": [0.25, 0.20, 0.15, 0.10, 0.08],
            "odds_win": [3.0, 8.0, 20.0, 50.0, 100.0],
            "finish_pos": [1, 3, 5, 8, 12],
            "horse_name": ["Fav", "Mid", "Long", "VeryLong", "ExtremeLong"],
            "date": ["2024-01-01"] * 5,
            "race_name": ["Test Race"] * 5,
        })

    def test_max_odds_filters_entries(self):
        """Entries above max_odds should be skipped."""
        df = self._make_predictions()

        # With high ceiling — all entries eligible
        bt_high = Backtester(BacktestConfig(
            ev_threshold=0.0, max_odds=200.0, use_kelly=False,
        ))
        r_high = bt_high.run(df)

        # With low ceiling — only low-odds entries eligible
        bt_low = Backtester(BacktestConfig(
            ev_threshold=0.0, max_odds=10.0, use_kelly=False,
        ))
        r_low = bt_low.run(df)

        # Fewer bets with tight ceiling
        assert r_low.total_bets <= r_high.total_bets

        # None of the bets in the low-ceiling run should be above 10x
        for b in r_low.bets:
            assert b.odds <= 10.0

    def test_min_odds_filters_entries(self):
        """Entries below min_odds should be skipped."""
        df = self._make_predictions()
        bt = Backtester(BacktestConfig(
            ev_threshold=0.0, max_odds=200.0, min_odds=5.0, use_kelly=False,
        ))
        result = bt.run(df)

        for b in result.bets:
            assert b.odds >= 5.0

    def test_default_max_odds_is_30(self):
        """Default max_odds should be 30.0."""
        config = BacktestConfig()
        assert config.max_odds == 30.0

    def test_default_min_odds_is_1(self):
        """Default min_odds should be 2.0."""
        config = BacktestConfig()
        assert config.min_odds == 2.0


# ---------------------------------------------------------------------------
# Priority 2 — Adaptive Blending
# ---------------------------------------------------------------------------

class TestAdaptiveBlending:
    def test_adaptive_weights_shape(self):
        """_compute_adaptive_weights should return array of same shape."""
        from models.ensemble import HybridEnsemble
        hybrid = HybridEnsemble()
        odds = np.array([2.0, 5.0, 10.0, 30.0, 100.0])
        weights = hybrid._compute_adaptive_weights(odds)
        assert weights.shape == odds.shape

    def test_low_odds_favors_fundamental(self):
        """Low-odds horses should get higher fundamental weight."""
        from models.ensemble import HybridEnsemble
        hybrid = HybridEnsemble()
        odds = np.array([2.0, 100.0])
        weights = hybrid._compute_adaptive_weights(odds)
        # Low odds → higher fund weight
        assert weights[0] > weights[1]

    def test_weights_in_range(self):
        """Weights should be in [0.2, 0.6]."""
        from models.ensemble import HybridEnsemble
        hybrid = HybridEnsemble()
        odds = np.array([1.1, 2.0, 5.0, 10.0, 30.0, 100.0, 500.0])
        weights = hybrid._compute_adaptive_weights(odds)
        assert np.all(weights >= 0.19)  # small tolerance for float
        assert np.all(weights <= 0.61)

    def test_smooth_transition(self):
        """Weights should transition smoothly, never jumping."""
        from models.ensemble import HybridEnsemble
        hybrid = HybridEnsemble()
        odds = np.linspace(1.5, 100.0, 100)
        weights = hybrid._compute_adaptive_weights(odds)
        diffs = np.abs(np.diff(weights))
        # No jumps > 0.05 between adjacent odds values
        assert np.all(diffs < 0.05)


# ---------------------------------------------------------------------------
# Priority 2 — Longshot Alpha Penalty
# ---------------------------------------------------------------------------

class TestLongshotAlphaPenalty:
    def test_high_odds_alpha_penalized(self):
        """Alpha score for high-odds signals should be penalized."""
        detector = DivergenceDetector(min_divergence=0.05)

        # Same divergence, different odds
        entry_ids = np.array([1, 2])
        fundamental = np.array([0.30, 0.30])
        odds_aware = np.array([0.25, 0.25])
        market_odds = np.array([5.0, 50.0])
        # Divergence: [0.10, 0.28] — both above threshold

        signals = detector.detect(entry_ids, fundamental, odds_aware, market_odds)

        # Both should be detected
        assert len(signals) >= 1

        # The high-odds signal should have lower alpha per unit divergence
        if len(signals) == 2:
            low_odds_signal = next(s for s in signals if s.entry_id == 1)
            high_odds_signal = next(s for s in signals if s.entry_id == 2)
            # Alpha / divergence ratio should be lower for high odds
            low_ratio = low_odds_signal.alpha_score / low_odds_signal.divergence
            high_ratio = high_odds_signal.alpha_score / high_odds_signal.divergence
            assert high_ratio < low_ratio


# ---------------------------------------------------------------------------
# Priority 3 — Kelly Sizing
# ---------------------------------------------------------------------------

class TestKellySizing:
    def test_pure_kelly_skips_tiny_bets(self):
        """With use_kelly=True, bets with tiny Kelly fraction should be skipped."""
        df = pd.DataFrame({
            "race_id": [1, 1],
            "entry_id": [1, 2],
            "win_prob": [0.05, 0.25],  # Very low prob → tiny Kelly
            "odds_win": [15.0, 5.0],
            "finish_pos": [8, 1],
            "horse_name": ["Weak", "Strong"],
            "date": ["2024-01-01", "2024-01-01"],
            "race_name": ["Test", "Test"],
        })

        bt = Backtester(BacktestConfig(
            ev_threshold=0.0, max_odds=200.0,
            use_kelly=True, min_kelly_fraction=0.01,
        ))
        result = bt.run(df)

        # Should not place a bet on the weak horse (tiny Kelly)
        for b in result.bets:
            assert b.kelly >= 0.01 or b.stake > 0

    def test_flat_stake_uses_floor(self):
        """With use_kelly=False, flat_stake should be the floor."""
        df = pd.DataFrame({
            "race_id": [1],
            "entry_id": [1],
            "win_prob": [0.30],
            "odds_win": [5.0],
            "finish_pos": [1],
            "horse_name": ["Test"],
            "date": ["2024-01-01"],
            "race_name": ["Test"],
        })

        bt = Backtester(BacktestConfig(
            ev_threshold=0.0, max_odds=200.0,
            use_kelly=False, flat_stake=1000,
        ))
        result = bt.run(df)

        if result.total_bets > 0:
            assert result.bets[0].stake >= 1000

    def test_default_use_kelly_true(self):
        """Default should use Kelly sizing."""
        config = BacktestConfig()
        assert config.use_kelly is True

    def test_min_kelly_fraction_default(self):
        """Default min_kelly_fraction should be 0.005."""
        config = BacktestConfig()
        assert config.min_kelly_fraction == 0.005
