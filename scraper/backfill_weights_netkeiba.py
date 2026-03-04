"""
UmaEdge — Backfill Horse IDs + Weights from Netkeiba Race Pages

For races scraped from JRA EN that have synthetic horse IDs (jra_*),
fetch the same race from netkeiba to:
  1. Map jra_* horse IDs → real netkeiba horse IDs
  2. Backfill horse_weight from the weight column

Match by race.netkeiba_id (our synthetic IDs follow netkeiba's format)
and entry.post_position within each race.

Usage:
    python -m scraper.backfill_weights_netkeiba                # backfill all
    python -m scraper.backfill_weights_netkeiba --limit 50     # first 50 races
    python -m scraper.backfill_weights_netkeiba --dry-run      # preview only
    python -m scraper.backfill_weights_netkeiba --workers 3    # concurrency
"""

import argparse
import logging
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

from scraper.db import get_session

log = logging.getLogger("backfill_weights_nk")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
log.addHandler(_handler)

RACE_URL = "https://db.netkeiba.com/race/{race_id}/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _fetch(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch a netkeiba URL (EUC-JP encoded) with retries."""
    for attempt in range(retries):
        try:
            delay = random.uniform(0.8, 1.5)
            time.sleep(delay)
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                html = resp.content.decode("euc-jp", errors="replace")
                return BeautifulSoup(html, "lxml")
            elif resp.status_code == 404:
                return None
            elif resp.status_code in (429, 503):
                backoff = min(60, 10 * (2 ** attempt))
                log.warning(f"Rate limited ({resp.status_code}), waiting {backoff}s")
                time.sleep(backoff)
            else:
                log.warning(f"HTTP {resp.status_code} for {url} (attempt {attempt + 1})")
        except requests.RequestException as e:
            log.warning(f"Request error: {e} (attempt {attempt + 1})")
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_weight(text_val: str) -> Optional[int]:
    """Parse weight like '466(+8)' → 466."""
    m = re.match(r"(\d{3,4})", text_val.strip())
    return int(m.group(1)) if m else None


def parse_race_result(soup: BeautifulSoup) -> list[dict]:
    """
    Parse a netkeiba race result page.

    Returns list of dicts:
      {"post_position": 10, "horse_code": "2018106541",
       "horse_name_jp": "ラストサムライ", "weight": 466}
    """
    results = []
    table = soup.find("table", class_="race_table_01")
    if not table:
        return results

    rows = table.find_all("tr")[1:]  # skip header
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 15:
            continue
        try:
            # Post position (column 2, 0-indexed)
            pp_text = cells[2].get_text(strip=True)
            if not pp_text.isdigit():
                continue
            post_pos = int(pp_text)

            # Horse link → netkeiba ID
            horse_link = cells[3].find("a", href=re.compile(r"/horse/"))
            if not horse_link:
                continue
            horse_href = horse_link.get("href", "")
            horse_code_m = re.search(r"/horse/(\d+)", horse_href)
            if not horse_code_m:
                continue
            horse_code = horse_code_m.group(1)
            horse_name_jp = horse_link.get_text(strip=True)

            # Horse weight (column 14, 0-indexed)
            weight_text = cells[14].get_text(strip=True)
            weight = _parse_weight(weight_text) if weight_text else None

            results.append({
                "post_position": post_pos,
                "horse_code": horse_code,
                "horse_name_jp": horse_name_jp,
                "weight": weight,
            })
        except Exception as e:
            log.debug(f"Error parsing row: {e}")
            continue

    return results


# ---------------------------------------------------------------------------
# DB Queries
# ---------------------------------------------------------------------------

def get_races_to_backfill(limit: Optional[int] = None) -> list[dict]:
    """
    Get races that have entries with jra_* horse IDs and missing weights.
    Returns list of {"race_db_id": ..., "netkeiba_id": ...}
    """
    query = """
        SELECT DISTINCT r.id AS race_db_id, r.netkeiba_id
        FROM races r
        JOIN entries e ON e.race_id = r.id
        JOIN horses h ON h.id = e.horse_id
        WHERE h.netkeiba_id LIKE 'jra_%%'
          AND e.horse_weight IS NULL
          AND r.netkeiba_id IS NOT NULL
        ORDER BY r.id
    """
    if limit:
        query += f" LIMIT {limit}"

    with get_session() as session:
        rows = session.execute(text(query)).fetchall()
    return [{"race_db_id": r[0], "netkeiba_id": r[1]} for r in rows]


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _process_one_race(race: dict, idx: int, total: int,
                      dry_run: bool) -> dict:
    """Fetch one netkeiba race page and update matching entries."""
    stats = {"ids_mapped": 0, "weights_set": 0, "merged": 0, "failed": 0}
    nk_race_id = race["netkeiba_id"]
    race_db_id = race["race_db_id"]

    url = RACE_URL.format(race_id=nk_race_id)
    soup = _fetch(url)
    if not soup:
        stats["failed"] = 1
        return stats

    parsed = parse_race_result(soup)
    if not parsed:
        stats["failed"] = 1
        return stats

    if dry_run:
        for p in parsed[:3]:
            log.info(
                f"  [{idx}/{total}] R{nk_race_id} PP={p['post_position']} "
                f"{p['horse_name_jp']} → code={p['horse_code']} wt={p['weight']}"
            )
        return stats

    with get_session() as session:
        for p in parsed:
            pp = p["post_position"]
            horse_code = p["horse_code"]
            weight = p["weight"]
            horse_name_jp = p["horse_name_jp"]

            # Find the entry in our DB for this race + post position
            entry_row = session.execute(text("""
                SELECT e.id AS entry_id, e.horse_id, h.netkeiba_id
                FROM entries e
                JOIN horses h ON h.id = e.horse_id
                WHERE e.race_id = :race_id
                  AND e.post_position = :pp
            """), {"race_id": race_db_id, "pp": pp}).fetchone()

            if not entry_row:
                continue

            entry_id = entry_row[0]
            jra_horse_id = entry_row[1]
            current_nk_id = entry_row[2]

            # Only process entries with jra_* horse IDs
            if current_nk_id and current_nk_id.startswith("jra_"):
                # Check if a horse with this real netkeiba_id already exists
                existing = session.execute(text("""
                    SELECT id FROM horses
                    WHERE netkeiba_id = :code
                """), {"code": horse_code}).fetchone()

                if existing:
                    # Horse already exists — merge: re-point entry to real horse
                    real_horse_id = existing[0]
                    session.execute(text("""
                        UPDATE entries SET horse_id = :real_hid
                        WHERE id = :eid
                    """), {"real_hid": real_horse_id, "eid": entry_id})
                    stats["merged"] += 1
                else:
                    # No duplicate — safe to rename jra_* → real netkeiba_id
                    session.execute(text("""
                        UPDATE horses
                        SET netkeiba_id = :nk_id, name_jp = :name_jp
                        WHERE id = :hid AND netkeiba_id LIKE 'jra_%%'
                    """), {
                        "nk_id": horse_code,
                        "name_jp": horse_name_jp,
                        "hid": jra_horse_id,
                    })
                stats["ids_mapped"] += 1

            # Update horse_weight on entry
            if weight:
                session.execute(text("""
                    UPDATE entries SET horse_weight = :wt
                    WHERE id = :eid AND horse_weight IS NULL
                """), {"wt": weight, "eid": entry_id})
                stats["weights_set"] += 1

    return stats


def backfill(limit: Optional[int] = None, dry_run: bool = False,
             workers: int = 3):
    """Main backfill: fetch netkeiba race pages, update IDs + weights."""
    races = get_races_to_backfill(limit=limit)
    total = len(races)
    log.info(f"Found {total} races to process (workers={workers})")

    if not races:
        log.info("Nothing to backfill!")
        return

    totals = {"ids_mapped": 0, "weights_set": 0, "merged": 0, "failed": 0, "done": 0}
    lock = threading.Lock()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one_race, race, i, total, dry_run): i
            for i, race in enumerate(races, 1)
        }
        for future in as_completed(futures):
            r = future.result()
            with lock:
                totals["ids_mapped"] += r["ids_mapped"]
                totals["weights_set"] += r["weights_set"]
                totals["merged"] += r["merged"]
                totals["failed"] += r["failed"]
                totals["done"] += 1
                done = totals["done"]
                if done % 50 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    log.info(
                        f"  [{done:,}/{total:,}] "
                        f"🔗 {totals['ids_mapped']:,} IDs "
                        f"(🔀 {totals['merged']:,} merged) | "
                        f"⚖️  {totals['weights_set']:,} weights | "
                        f"❌ {totals['failed']} | "
                        f"{rate:.1f} races/s | ETA {eta:.0f}s"
                    )

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 60}")
    log.info(
        f"Done in {elapsed:.0f}s: "
        f"{totals['ids_mapped']:,} horse IDs mapped "
        f"({totals['merged']:,} merged), "
        f"{totals['weights_set']:,} weights set, "
        f"{totals['failed']} failed"
    )
    log.info(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UmaEdge — Backfill horse IDs + weights from netkeiba"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max races to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse only, don't update DB")
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent workers (default: 3, be polite to netkeiba)")
    args = parser.parse_args()
    backfill(limit=args.limit, dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()
