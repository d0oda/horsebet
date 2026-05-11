"""
Unit tests for the hybrid ensemble and divergence detector.
Sprint 2.2 — tests divergence detection and hybrid ensemble output shapes.
"""

import numpy as np
import pytest

from models.ensemble import DivergenceDetector, AlphaSignal


# ---------------------------------------------------------------------------
# Divergence Detector
# ---------------------------------------------------------------------------

class TestDivergenceDetector:
    def test_detects_positive_divergence(self):
        """When fundamental prob > market implied prob by threshold, detect signal."""
        detector = DivergenceDetector(min_divergence=0.05)

        entry_ids = np.array([1, 2, 3])
        fundamental = np.array([0.30, 0.16, 0.05])
        odds_aware = np.array([0.25, 0.13, 0.04])
        market_odds = np.array([5.0, 10.0, 50.0])
        # Market implied: [0.20, 0.10, 0.02]
        # Divergence: [0.10, 0.06, 0.03]

        signals = detector.detect(entry_ids, fundamental, odds_aware, market_odds)

        # Entry 1 (div=0.10) and entry 2 (div=0.05) should be detected
        assert len(signals) == 2
        assert signals[0].entry_id == 1  # highest alpha
        assert signals[0].divergence == pytest.approx(0.10, abs=0.01)
        assert signals[1].entry_id == 2

    def test_no_signal_when_below_threshold(self):
        """No signals when all divergences are below the min_divergence."""
        detector = DivergenceDetector(min_divergence=0.10)

        entry_ids = np.array([1, 2])
        fundamental = np.array([0.12, 0.11])
        odds_aware = np.array([0.10, 0.09])
        market_odds = np.array([10.0, 10.0])
        # Market implied: [0.10, 0.10]
        # Divergence: [0.02, 0.01] — both below 0.10

        signals = detector.detect(entry_ids, fundamental, odds_aware, market_odds)
        assert len(signals) == 0

    def test_no_signal_when_models_agree_with_market(self):
        """When fundamental ≈ market, no alpha signal."""
        detector = DivergenceDetector(min_divergence=0.05)

        entry_ids = np.array([1, 2])
        fundamental = np.array([0.20, 0.10])
        odds_aware = np.array([0.20, 0.10])
        market_odds = np.array([5.0, 10.0])
        # Market implied: [0.20, 0.10]
        # Divergence: [0.00, 0.00]

        signals = detector.detect(entry_ids, fundamental, odds_aware, market_odds)
        assert len(signals) == 0

    def test_handles_zero_odds(self):
        """Entries with zero or invalid odds are skipped."""
        detector = DivergenceDetector(min_divergence=0.05)

        entry_ids = np.array([1, 2])
        fundamental = np.array([0.30, 0.25])
        odds_aware = np.array([0.25, 0.20])
        market_odds = np.array([0.0, -1.0])

        signals = detector.detect(entry_ids, fundamental, odds_aware, market_odds)
        assert len(signals) == 0

    def test_signals_sorted_by_alpha_score(self):
        """Signals should be returned sorted by alpha_score descending."""
        detector = DivergenceDetector(min_divergence=0.05)

        entry_ids = np.array([1, 2, 3])
        fundamental = np.array([0.30, 0.45, 0.35])
        odds_aware = np.array([0.25, 0.40, 0.30])
        market_odds = np.array([5.0, 3.0, 4.0])
        # Market implied: [0.20, 0.333, 0.25]
        # Divergence: [0.10, 0.117, 0.10]

        signals = detector.detect(entry_ids, fundamental, odds_aware, market_odds)
        assert len(signals) >= 2

        # Should be sorted descending
        for i in range(len(signals) - 1):
            assert signals[i].alpha_score >= signals[i + 1].alpha_score

    def test_alpha_signal_fields(self):
        """AlphaSignal dataclass has correct fields."""
        signal = AlphaSignal(
            entry_id=42,
            fundamental_prob=0.25,
            market_prob=0.15,
            odds_aware_prob=0.22,
            divergence=0.10,
            alpha_score=0.08,
        )
        assert signal.entry_id == 42
        assert signal.fundamental_prob == 0.25
        assert signal.divergence == 0.10


# ---------------------------------------------------------------------------
# ODDS_FEATURES constant
# ---------------------------------------------------------------------------

class TestOddsFeatures:
    def test_odds_features_list(self):
        """ODDS_FEATURES should contain all odds-derived feature names."""
        from models.features import ODDS_FEATURES

        assert "odds_win" in ODDS_FEATURES
        assert "log_odds" in ODDS_FEATURES
        assert "popularity" in ODDS_FEATURES
        assert "odds_win_z" in ODDS_FEATURES
        assert "log_odds_z" in ODDS_FEATURES
        assert "popularity_z" in ODDS_FEATURES

    def test_odds_features_length(self):
        from models.features import ODDS_FEATURES
        assert len(ODDS_FEATURES) == 16
