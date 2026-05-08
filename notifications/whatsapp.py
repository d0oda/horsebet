"""
UmaEdge — WhatsApp Notification Backend (via CallMeBot).

Sends notifications via the CallMeBot free WhatsApp API.
Requires WHATSAPP_PHONE and WHATSAPP_API_KEY environment variables.

Setup (one-time):
  1. Add the CallMeBot phone number to your contacts:
       +34 694 242 562
  2. Send this exact message to them on WhatsApp:
       "I allow callmebot to send me messages"
  3. You'll receive an API key — put it in your .env file.

Usage:
    from notifications.whatsapp import WhatsAppNotifier
    notifier = WhatsAppNotifier()
    notifier.send_message("Hello from UmaEdge!")
"""

import logging
import os
from urllib.parse import quote

import requests

log = logging.getLogger("notifications.whatsapp")

CALLMEBOT_URL = "https://api.callmebot.com/whatsapp.php"


class WhatsAppNotifier:
    """Send notifications via CallMeBot WhatsApp API."""

    def __init__(self):
        self.phone = os.getenv("WHATSAPP_PHONE", "")
        self.api_key = os.getenv("WHATSAPP_API_KEY", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.phone and self.api_key)

    def send_message(self, text: str) -> bool:
        """Send a text message via WhatsApp."""
        if not self.is_configured:
            log.debug("WhatsApp not configured, skipping")
            return False

        try:
            resp = requests.get(
                CALLMEBOT_URL,
                params={
                    "phone": self.phone,
                    "text": text,
                    "apikey": self.api_key,
                },
                timeout=15,
            )

            if resp.status_code == 200:
                log.info("📨 WhatsApp message sent")
                return True
            else:
                log.warning(f"CallMeBot API error: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            log.error(f"WhatsApp send failed: {e}")
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
