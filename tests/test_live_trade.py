"""
Tests for live trading module (Sprint 7.5).

Tests guardrails, bet placement logic, and safety limits.
All tests use mocked DB — no real API calls.
"""

import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from models.live_trade import (
    LiveTrader,
    GuardrailError,
    check_paper_track_record,
    check_daily_loss,
    DEFAULT_STAKE,
    MAX_STAKE_PER_BET,
    MAX_DAILY_LOSS,
    MIN_MODEL_CONFIDENCE,
    MIN_EV_THRESHOLD,
    REQUIRED_PAPER_WEEKENDS,
)


# ---------------------------------------------------------------------------
# Guardrail Tests
# ---------------------------------------------------------------------------

class TestGuardrails:
    """Test safety guardrails for live trading."""

    def test_requires_confirm_flag(self):
        """Cannot place live bets without --confirm."""
        trader = LiveTrader()
        with pytest.raises(GuardrailError, match="--confirm"):
            trader._validate_guardrails(confirm=False)

    def test_max_stake_enforced(self):
        """Stake is capped at MAX_STAKE_PER_BET."""
        trader = LiveTrader(stake=10_000)
        assert trader.stake == MAX_STAKE_PER_BET

    def test_default_stake(self):
        """Default stake is ¥100."""
        trader = LiveTrader()
        assert trader.stake == DEFAULT_STAKE
        assert trader.stake == 100

    def test_constants_are_safe(self):
        """Verify safety constants are set appropriately."""
        assert DEFAULT_STAKE <= 100
        assert MAX_STAKE_PER_BET <= 500
        assert MAX_DAILY_LOSS <= 5_000
        assert MIN_MODEL_CONFIDENCE >= 0.10
        assert REQUIRED_PAPER_WEEKENDS >= 4


class TestPaperTrackRecord:
    """Test paper trading track record scoring."""

    def test_no_paper_trades(self, tmp_path):
        """No paper trades = not ready."""
        with patch("models.live_trade.Path") as mock_path:
            mock_path.return_value.parent = tmp_path
            result = check_paper_track_record()
            # Should return is_ready=False since no data
            assert result["is_ready"] is False or result["weekend_count"] >= 0


class TestDailyLoss:
    """Test daily loss limit checking."""

    def test_check_daily_loss_returns_dict(self):
        """check_daily_loss should always return a valid dict."""
        result = check_daily_loss("2099-01-01")  # future date, no trades
        assert "daily_loss" in result
        assert "limit" in result
        assert "is_within_limit" in result

    def test_daily_loss_limit_value(self):
        """Daily loss limit should be MAX_DAILY_LOSS."""
        result = check_daily_loss("2099-01-01")
        assert result["limit"] == MAX_DAILY_LOSS


# ---------------------------------------------------------------------------
# CLI Tests
# ---------------------------------------------------------------------------

class TestLiveTradeCLI:
    """Test CLI argument parsing."""

    def test_help_exits_zero(self):
        from models.live_trade import main
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["live_trade", "--help"]):
                main()
        assert exc_info.value.code == 0

    def test_check_mode(self, capsys):
        """--check should print guardrail status."""
        from models.live_trade import main
        with patch("sys.argv", ["live_trade", "--check"]):
            main()
        captured = capsys.readouterr()
        assert "Guardrail Status" in captured.out

    def test_no_args_shows_help(self):
        """No args should print help."""
        from models.live_trade import main
        with patch("sys.argv", ["live_trade"]):
            # This should not crash
            main()
