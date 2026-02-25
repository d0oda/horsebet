"""
Tests for Sprint 3 & 4 modules:
  - analyse_bets.py (failure mode classification)
  - backtest_trio.py (trio probability computation)
  - drift.py (metrics logging)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from models.analyse_bets import classify_bet, FailureMode, analyse_losing_bets
from models.backtest_trio import compute_trio_prob, run_trio_backtest, TrioBacktestResult


# ---------------------------------------------------------------------------
# Analyse Bets — Failure Mode Classification
# ---------------------------------------------------------------------------

class TestClassifyBet:
    def test_variance_finish_2nd(self):
        assert classify_bet(0.15, 0.10, 2) == "variance"

    def test_variance_finish_3rd(self):
        assert classify_bet(0.15, 0.10, 3) == "variance"

    def test_overconfident(self):
        """Model prob > 2x market prob."""
        assert classify_bet(0.30, 0.10, 5) == "overconfident"

    def test_no_edge(self):
        """Model prob close to market prob (ratio < 1.5)."""
        assert classify_bet(0.12, 0.10, 6) == "no_edge"

    def test_marginal(self):
        """Ratio between 1.5 and 2.0 = unknown/marginal."""
        assert classify_bet(0.18, 0.10, 7) == "unknown"

    def test_zero_market_prob(self):
        """Zero market prob should not crash."""
        result = classify_bet(0.15, 0.0, 5)
        assert result == "overconfident"


# ---------------------------------------------------------------------------
# Backtest Trio — Probability Computation
# ---------------------------------------------------------------------------

class TestTrioProb:
    def test_basic_trio_prob(self):
        # Three horses with 20% win prob each
        probs = {1: 0.20, 2: 0.20, 3: 0.20}
        p = compute_trio_prob(probs, (1, 2, 3))
        assert 0 < p < 1

    def test_low_prob_horses(self):
        probs = {1: 0.01, 2: 0.01, 3: 0.01}
        p = compute_trio_prob(probs, (1, 2, 3))
        assert p > 0

    def test_high_prob_trio(self):
        """Top 3 favourites should have higher trio probability."""
        probs = {1: 0.30, 2: 0.25, 3: 0.20, 4: 0.05, 5: 0.05}
        p_top = compute_trio_prob(probs, (1, 2, 3))
        p_bottom = compute_trio_prob(probs, (3, 4, 5))
        assert p_top > p_bottom


class TestTrioBacktest:
    def test_empty_df(self):
        df = pd.DataFrame({"race_id": [], "entry_id": [], "horse_id": [],
                           "win_prob": [], "odds_win": [], "finish_pos": [],
                           "date": [], "horse_name": []})
        result = run_trio_backtest(df)
        assert result.total_tickets == 0

    def test_basic_backtest(self):
        # Create a simple race with known results
        df = pd.DataFrame({
            "race_id": [1] * 5,
            "entry_id": list(range(5)),
            "horse_id": list(range(5)),
            "win_prob": [0.30, 0.25, 0.20, 0.15, 0.10],
            "odds_win": [3.0, 4.0, 5.0, 7.0, 10.0],
            "finish_pos": [1, 2, 3, 4, 5],
            "date": ["2024-01-01"] * 5,
            "horse_name": ["A", "B", "C", "D", "E"],
        })
        result = run_trio_backtest(df, min_ev=-1.0)  # allow all tickets
        assert isinstance(result, TrioBacktestResult)
        assert result.total_races == 1
        assert result.total_tickets > 0


# ---------------------------------------------------------------------------
# Drift Tracker
# ---------------------------------------------------------------------------

class TestDriftTracker:
    def test_log_and_read(self):
        import models.drift as drift_mod

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = Path(f.name)

        # Patch the log path
        with patch.object(drift_mod, "DRIFT_LOG_PATH", tmp_path):
            drift_mod.log_metrics(
                version="test_v1", auc=0.75, log_loss=0.45, brier=0.18, n_samples=100
            )
            drift_mod.log_metrics(
                version="test_v2", auc=0.77, log_loss=0.43, brier=0.16, n_samples=120
            )

            summary = drift_mod.get_drift_summary()

        assert summary["status"] == "ok"
        assert summary["n_versions"] == 2
        assert summary["latest"]["version"] == "test_v2"
        assert summary["trend"]["direction"] == "improving"
        assert summary["trend"]["auc_change"] == pytest.approx(0.02, abs=0.001)

        # Cleanup
        tmp_path.unlink()
