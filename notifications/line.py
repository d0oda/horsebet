"""
UmaEdge — LINE Notification Backend.

Sends notifications via the LINE Messaging API (push messages).
Requires LINE_CHANNEL_TOKEN and LINE_USER_ID environment variables.

Usage:
    from notifications.line import LineNotifier
    notifier = LineNotifier()
    notifier.send_message("Hello from UmaEdge!")
"""

import logging
import os

import requests

log = logging.getLogger("notifications.line")

LINE_API = "https://api.line.me/v2/bot/message/push"


class LineNotifier:
    """Send notifications via LINE Messaging API."""

    def __init__(self):
        self.channel_token = os.getenv("LINE_CHANNEL_TOKEN", "")
        self.user_id = os.getenv("LINE_USER_ID", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.channel_token and self.user_id)

    def send_message(self, text: str) -> bool:
        """Send a push message to the configured LINE user."""
        if not self.is_configured:
            log.debug("LINE not configured, skipping")
            return False

        try:
            resp = requests.post(
                LINE_API,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.channel_token}",
                },
                json={
                    "to": self.user_id,
                    "messages": [{"type": "text", "text": text}],
                },
                timeout=10,
            )

            if resp.status_code == 200:
                log.info("📨 LINE message sent")
                return True
            else:
                log.warning(f"LINE API error: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            log.error(f"LINE send failed: {e}")
            return False

    def send_bet_alert(self, bet: dict) -> bool:
        """Send a formatted bet notification."""
        emoji = "🔴" if bet.get("live") else "📝"
        mode = "LIVE" if bet.get("live") else "Paper"

        text = (
            f"{emoji} {mode} Bet Placed\n"
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
            f"{emoji} UmaEdge Daily Summary\n"
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
            f"✅ Trades Reconciled\n"
            f"Settled: {summary.get('reconciled', 0)} trades\n"
            f"Total P&L: ¥{summary.get('total_profit', 0):+,}"
        )
        return self.send_message(text)
