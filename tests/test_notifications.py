"""
Tests for notification backends and dispatcher.

All tests use mocked HTTP — no real API calls or tokens needed.
"""

from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Telegram Backend
# ---------------------------------------------------------------------------

class TestTelegramNotifier:
    """Test Telegram notification formatting and sending."""

    def test_not_configured_without_token(self):
        """Without env vars, Telegram should not be configured."""
        with patch.dict("os.environ", {}, clear=True):
            from notifications.telegram import TelegramNotifier
            notifier = TelegramNotifier()
            assert notifier.is_configured is False

    def test_configured_with_token(self):
        """With both env vars, Telegram should be configured."""
        with patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_CHAT_ID": "123",
        }):
            from notifications.telegram import TelegramNotifier
            notifier = TelegramNotifier()
            assert notifier.is_configured is True

    def test_send_skips_when_not_configured(self):
        """Sending should silently skip when not configured."""
        with patch.dict("os.environ", {}, clear=True):
            from notifications.telegram import TelegramNotifier
            notifier = TelegramNotifier()
            result = notifier.send_message("test")
            assert result is False

    @patch("notifications.telegram.requests.post")
    def test_send_message_success(self, mock_post):
        """Test successful message send."""
        mock_post.return_value = MagicMock(status_code=200)
        with patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_CHAT_ID": "123",
        }):
            from notifications.telegram import TelegramNotifier
            notifier = TelegramNotifier()
            result = notifier.send_message("test message")
            assert result is True
            mock_post.assert_called_once()

    @patch("notifications.telegram.requests.post")
    def test_bet_alert_formatting(self, mock_post):
        """Test bet alert message formatting."""
        mock_post.return_value = MagicMock(status_code=200)
        with patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_CHAT_ID": "123",
        }):
            from notifications.telegram import TelegramNotifier
            notifier = TelegramNotifier()
            notifier.send_bet_alert({
                "horse_name": "Deep Impact",
                "model_prob": 0.25,
                "odds": 3.5,
                "ev": 0.12,
                "stake": 1000,
                "live": False,
            })

            call_args = mock_post.call_args
            text = call_args.kwargs["json"]["text"] if "json" in call_args.kwargs else call_args[1]["json"]["text"]
            assert "Deep Impact" in text
            assert "Paper" in text


# ---------------------------------------------------------------------------
# LINE Backend
# ---------------------------------------------------------------------------

class TestLineNotifier:
    """Test LINE notification formatting and sending."""

    def test_not_configured_without_token(self):
        """Without env vars, LINE should not be configured."""
        with patch.dict("os.environ", {}, clear=True):
            from notifications.line import LineNotifier
            notifier = LineNotifier()
            assert notifier.is_configured is False

    def test_configured_with_token(self):
        """With both env vars, LINE should be configured."""
        with patch.dict("os.environ", {
            "LINE_CHANNEL_TOKEN": "test-token",
            "LINE_USER_ID": "U123",
        }):
            from notifications.line import LineNotifier
            notifier = LineNotifier()
            assert notifier.is_configured is True

    @patch("notifications.line.requests.post")
    def test_send_message_success(self, mock_post):
        """Test successful message send."""
        mock_post.return_value = MagicMock(status_code=200)
        with patch.dict("os.environ", {
            "LINE_CHANNEL_TOKEN": "test-token",
            "LINE_USER_ID": "U123",
        }):
            from notifications.line import LineNotifier
            notifier = LineNotifier()
            result = notifier.send_message("test")
            assert result is True


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class TestDispatcher:
    """Test notification dispatcher routing."""

    def test_no_backends_returns_zero(self):
        """With no backends configured, dispatcher sends 0."""
        with patch.dict("os.environ", {}, clear=True):
            from notifications.dispatcher import notify_message
            result = notify_message("test")
            assert result == 0

    def test_get_status_structure(self):
        """get_status should return expected keys."""
        with patch.dict("os.environ", {}, clear=True):
            from notifications.dispatcher import get_status
            status = get_status()
            assert "telegram" in status
            assert "line" in status
            assert "any_configured" in status
            assert isinstance(status["any_configured"], bool)

    @patch("notifications.telegram.requests.post")
    def test_dispatch_to_telegram(self, mock_post):
        """When Telegram is configured, messages are dispatched."""
        mock_post.return_value = MagicMock(status_code=200)
        with patch.dict("os.environ", {
            "TELEGRAM_BOT_TOKEN": "test",
            "TELEGRAM_CHAT_ID": "123",
        }, clear=True):
            from notifications.dispatcher import notify_message
            result = notify_message("test")
            assert result >= 1
