"""
UmaEdge — Backfill Exotic Dividends from netkeiba.

Scrapes actual JRA exotic payouts (三連複, 三連単, 馬連, 馬単, ワイド)
from db.netkeiba.com result pages and stores them in the dividends table.

Usage:
    # Backfill all races in the DB that don't have dividends yet
    python -m scraper.backfill_dividends

    # Backfill a specific date range
    python -m scraper.backfill_dividends --start 2023-01-01 --end 2023-12-31

    # Dry run (parse but don't save)
    python -m scraper.backfill_dividends --dry-run --limit 5
"""

import argparse
import logging
import re
import time
import random
import os

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

from scraper.db import get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_dividends")

LEGACY_RACE_URL = "https://db.netkeiba.com/race/{race_id}/"
DELAY_MIN = float(os.getenv("SCRAPE_DELAY_MIN", 2))
DELAY_MAX = float(os.getenv("SCRAPE_DELAY_MAX", 4))
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

# Map Japanese bet type names to our internal names
BET_TYPE_MAP = {
    "単勝": "win",
    "複勝": "place",
    "枠連": "bracket_quinella",
    "馬連": "quinella",
    "馬単": "exacta",
    "ワイド": "wide",
    "三連複": "trio",
    "三連単": "trifecta",
}


def _split_by_br(td) -> list[str]:
    """Split a <td> cell into parts separated by <br/> tags."""
    parts = []
    for child in td.children:
        if isinstance(child, str):
            t = child.strip()
            if t:
                parts.append(t)
        elif child.name == "br":
            continue
        else:
            parts.append(child.get_text(strip=True))
    return parts if parts else [td.get_text(strip=True)]


def parse_dividends(html: str) -> list[dict]:
    """
    Parse payout tables from a db.netkeiba.com race result page.
    Returns a list of dicts: {bet_type, combination, payout, popularity}
    """
    soup = BeautifulSoup(html, "html.parser")
    dividends = []

    for table in soup.find_all("table", class_="pay_table_01"):
        for tr in table.find_all("tr"):
            th = tr.find("th")
            tds = tr.find_all("td")
            if not th or len(tds) < 2:
                continue

            bet_type_jp = th.get_text(strip=True)
            bet_type = BET_TYPE_MAP.get(bet_type_jp)
            if not bet_type:
                continue

            # Split each cell by <br/> for multi-value rows (複勝, ワイド)
            combos = _split_by_br(tds[0])
            payouts = _split_by_br(tds[1])
            pops = _split_by_br(tds[2]) if len(tds) > 2 else []

            for i, combo_text in enumerate(combos):
                combo = _normalize_combination(combo_text, bet_type)
                payout = _parse_amount(payouts[i]) if i < len(payouts) else 0
                pop = None
                if i < len(pops):
                    try:
                        pop = int(pops[i].replace(",", ""))
                    except ValueError:
                        pass

                if combo and payout:
                    dividends.append({
                        "bet_type": bet_type,
                        "combination": combo,
                        "payout": payout,
                        "popularity": pop,
                    })

    return dividends


def _normalize_combination(text: str, bet_type: str) -> str:
    """
    Normalize a combination string to our internal format.
    e.g. "10 → 5 → 11" → "10-5-11" (ordered, for trifecta/exacta)
    e.g. "5 - 10 - 11" → "5-10-11" (sorted, for trio/quinella/wide)
    """
    # Remove all whitespace and arrows
    text = text.strip()
    
    # Extract just the numbers
    # Handle formats: "10 → 5 → 11", "5 - 10 - 11", "10→5", "5-10", plain "10"
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return ""

    if bet_type in ("trifecta", "exacta"):
        # Ordered combination (order matters)
        return "-".join(numbers)
    elif bet_type in ("trio", "quinella", "wide", "bracket_quinella"):
        # Unordered combination (sort for consistency)
        return "-".join(sorted(numbers, key=int))
    else:
        # win, place — just horse number
        return "-".join(numbers)


def _parse_amount(text: str) -> int:
    """Parse a payout amount like '44,030' → 44030 or '円' suffix."""
    text = text.replace(",", "").replace("円", "").replace("¥", "").strip()
    try:
        return int(text)
    except ValueError:
        return 0


def scrape_race_dividends(netkeiba_id: str) -> list[dict]:
    """Scrape dividends for a single race from db.netkeiba.com."""
    url = LEGACY_RACE_URL.format(race_id=netkeiba_id)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    
    if resp.status_code != 200:
        log.warning(f"HTTP {resp.status_code} for {netkeiba_id}")
        return []

    # db.netkeiba.com uses EUC-JP encoding
    resp.encoding = "euc-jp"
    dividends = parse_dividends(resp.text)
    return dividends


def save_dividends(race_id: int, dividends: list[dict], session):
    """Save dividends to the database."""
    for d in dividends:
        session.execute(text("""
            INSERT INTO dividends (race_id, bet_type, combination, payout, popularity)
            VALUES (:race_id, :bet_type, :combination, :payout, :popularity)
            ON CONFLICT (race_id, bet_type, combination) DO UPDATE SET
                payout = EXCLUDED.payout,
                popularity = EXCLUDED.popularity
        """), {
            "race_id": race_id,
            "bet_type": d["bet_type"],
            "combination": d["combination"],
            "payout": d["payout"],
            "popularity": d["popularity"],
        })
    session.commit()


def backfill(start_date=None, end_date=None, limit=None, dry_run=False):
    """Backfill dividends for all races missing them."""
    with get_session() as session:
        # Find races that have results but no dividends yet
        query = """
            SELECT r.id, r.netkeiba_id, r.date
            FROM races r
            JOIN entries e ON e.race_id = r.id
            JOIN results res ON res.entry_id = e.id
            WHERE r.netkeiba_id IS NOT NULL
              AND r.id NOT IN (SELECT DISTINCT race_id FROM dividends)
        """
        params = {}
        if start_date:
            query += " AND r.date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND r.date <= :end_date"
            params["end_date"] = end_date

        query += " GROUP BY r.id, r.netkeiba_id, r.date ORDER BY r.date DESC"

        if limit:
            query += f" LIMIT {limit}"

        races = session.execute(text(query), params).fetchall()
        log.info(f"Found {len(races)} races missing dividends")

        success = 0
        fail = 0
        for i, race in enumerate(races):
            race_id, netkeiba_id, date = race
            log.info(f"[{i+1}/{len(races)}] Scraping {netkeiba_id} ({date})...")

            try:
                dividends = scrape_race_dividends(netkeiba_id)
                if not dividends:
                    log.warning(f"  No dividends found for {netkeiba_id}")
                    fail += 1
                    continue

                # Filter to exotic types we care about
                exotic_types = {"trio", "trifecta", "quinella", "exacta", "wide"}
                exotic = [d for d in dividends if d["bet_type"] in exotic_types]

                if dry_run:
                    for d in dividends:
                        log.info(f"  {d['bet_type']:>12}: {d['combination']:>12} → ¥{d['payout']:>8,} (pop {d['popularity']})")
                else:
                    save_dividends(race_id, dividends, session)

                log.info(f"  ✅ {len(dividends)} dividends ({len(exotic)} exotic)")
                success += 1

            except Exception as e:
                log.error(f"  ❌ Error: {e}")
                fail += 1

            # Rate limiting
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        log.info(f"\nDone: {success} scraped, {fail} failed out of {len(races)} races")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill exotic dividends from netkeiba")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, help="Max races to scrape")
    parser.add_argument("--dry-run", action="store_true", help="Parse but don't save")
    args = parser.parse_args()

    backfill(
        start_date=args.start,
        end_date=args.end,
        limit=args.limit,
        dry_run=args.dry_run,
    )
