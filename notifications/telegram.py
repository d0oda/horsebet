"""
UmaEdge — Telegram Notification Backend.

Sends notifications via the Telegram Bot API.
Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.

Usage:
    from notifications.telegram import TelegramNotifier
    notifier = TelegramNotifier()
    notifier.send_message("Hello from UmaEdge!")
"""

import logging
import os

import requests

log = logging.getLogger("notifications.telegram")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Send notifications via Telegram Bot API."""

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a text message to the configured Telegram chat."""
        if not self.is_configured:
            log.debug("Telegram not configured, skipping")
            return False

        try:
            url = TELEGRAM_API.format(token=self.token)
            resp = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }, timeout=10)

            if resp.status_code == 200:
                log.info("📨 Telegram message sent")
                return True
            else:
                log.warning(f"Telegram API error: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            log.error(f"Telegram send failed: {e}")
            return False

    def send_bet_alert(self, bet: dict) -> bool:
        """Send a formatted bet notification."""
        emoji = "🔴" if bet.get("live") else "📝"
        mode = "LIVE" if bet.get("live") else "Paper"

        text = (
            f"{emoji} *{mode} Bet Placed*\n"
            f"🐴 {bet.get('horse_name', '?')}\n"
            f"📊 P(win): {bet.get('model_prob', 0):.1%}\n"
            f"💰 Odds: {bet.get('odds', 0):.1f}x\n"
            f"📈 EV: {bet.get('ev', 0):+.2f}\n"
            f"💵 Stake: ¥{bet.get('stake', 0):,}"
        )
        return self.send_message(text)

    def send_daily_summary(self, summary: dict) -> bool:
        """Send daily P&L summary."""
        profit = summary.get("total_profit", 0)
        emoji = "📈" if profit >= 0 else "📉"

        text = (
            f"{emoji} *UmaEdge Daily Summary*\n"
            f"Bets: {summary.get('total_bets', 0)}\n"
            f"Wins: {summary.get('wins', 0)}\n"
            f"Staked: ¥{summary.get('total_staked', 0):,}\n"
            f"Profit: ¥{profit:+,}\n"
            f"ROI: {summary.get('roi_pct', 0):+.1f}%"
        )
        return self.send_message(text)

    def send_reconciliation(self, summary: dict) -> bool:
        """Send reconciliation result."""
        text = (
            f"✅ *Trades Reconciled*\n"
            f"Settled: {summary.get('reconciled', 0)} trades\n"
            f"Total P&L: ¥{summary.get('total_profit', 0):+,}"
        )
        return self.send_message(text)
