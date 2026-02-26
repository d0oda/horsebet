"""
UmaEdge — Notification Dispatcher.

Unified dispatcher that routes notifications to all configured backends
(Telegram, LINE). Auto-detects which backends have tokens configured.

Usage:
    from notifications.dispatcher import notify_bet_placed, notify_reconciliation

    notify_bet_placed(bet_dict)
    notify_reconciliation(summary_dict)
"""

import logging
from typing import Optional

log = logging.getLogger("notifications.dispatcher")


def _get_backends() -> list:
    """Get all configured notification backends."""
    backends = []

    from notifications.telegram import TelegramNotifier
    tg = TelegramNotifier()
    if tg.is_configured:
        backends.append(tg)
        log.debug("Telegram backend active")

    from notifications.line import LineNotifier
    line = LineNotifier()
    if line.is_configured:
        backends.append(line)
        log.debug("LINE backend active")

    return backends


def notify_bet_placed(bet: dict) -> int:
    """
    Notify all configured backends about a placed bet.

    Args:
        bet: Dict with keys: horse_name, model_prob, odds, ev, stake, live (bool)

    Returns:
        Number of successful notifications sent.
    """
    backends = _get_backends()
    if not backends:
        log.debug("No notification backends configured")
        return 0

    sent = 0
    for backend in backends:
        try:
            if backend.send_bet_alert(bet):
                sent += 1
        except Exception as e:
            log.warning(f"Notification failed ({type(backend).__name__}): {e}")

    return sent


def notify_reconciliation(summary: dict) -> int:
    """
    Notify all configured backends about reconciliation results.

    Args:
        summary: Dict with keys: reconciled, total_profit

    Returns:
        Number of successful notifications sent.
    """
    backends = _get_backends()
    sent = 0
    for backend in backends:
        try:
            if backend.send_reconciliation(summary):
                sent += 1
        except Exception as e:
            log.warning(f"Notification failed ({type(backend).__name__}): {e}")

    return sent


def notify_daily_summary(summary: dict) -> int:
    """
    Notify all configured backends with daily P&L summary.

    Args:
        summary: Dict with keys: total_bets, wins, total_staked, total_profit, roi_pct

    Returns:
        Number of successful notifications sent.
    """
    backends = _get_backends()
    sent = 0
    for backend in backends:
        try:
            if backend.send_daily_summary(summary):
                sent += 1
        except Exception as e:
            log.warning(f"Notification failed ({type(backend).__name__}): {e}")

    return sent


def notify_message(text: str) -> int:
    """
    Send a plain text message to all configured backends.

    Returns:
        Number of successful notifications sent.
    """
    backends = _get_backends()
    sent = 0
    for backend in backends:
        try:
            if backend.send_message(text):
                sent += 1
        except Exception as e:
            log.warning(f"Notification failed ({type(backend).__name__}): {e}")

    return sent


def get_status() -> dict:
    """Get configuration status of all backends."""
    from notifications.telegram import TelegramNotifier
    from notifications.line import LineNotifier

    tg = TelegramNotifier()
    line = LineNotifier()

    return {
        "telegram": {"configured": tg.is_configured},
        "line": {"configured": line.is_configured},
        "any_configured": tg.is_configured or line.is_configured,
    }
