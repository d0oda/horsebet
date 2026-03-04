"""
UmaEdge — Backfill Horse Body Weights from Umanity.jp

The horse records page at umanity.jp includes per-race body weight in the
`HorseWt(Kg)` column (e.g. "478(+6)"). This backfills `entries.horse_weight`
for entries that are currently NULL.

URL format: https://umanity.jp/en/racedata/db/horse_top.php?code={netkeiba_id}

Usage:
    python -m scraper.backfill_weights_umanity                # backfill all
    python -m scraper.backfill_weights_umanity --limit 100    # first 100 horses
    python -m scraper.backfill_weights_umanity --dry-run      # preview only
    python -m scraper.backfill_weights_umanity --workers 4    # concurrency
    python -m scraper.backfill_weights_umanity --year 2021    # only 2021 entries
"""

import argparse
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

from scraper.db import get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backfill_weights")

HORSE_URL = "https://umanity.jp/en/racedata/db/horse_top.php?code={code}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _fetch(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch a URL with retries and polite delay."""
    for attempt in range(retries):
        try:
            delay = random.uniform(0.3, 0.8)
            time.sleep(delay)
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            elif resp.status_code == 404:
                return None
            elif resp.status_code in (429, 503):
                backoff = min(60, 10 * (2 ** attempt))
                log.warning(f"Rate limited ({resp.status_code}), backing off {backoff}s")
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

def _parse_weight_cell(text_val: str) -> Optional[int]:
    """
    Parse a Umanity HorseWt(Kg) cell like '478(+6)' or '472(--)'.
    Returns the base weight as an integer, or None.
    """
    if not text_val:
        return None
    m = re.match(r"(\d{3,4})", text_val.strip())
    if m:
        return int(m.group(1))
    return None


def _parse_date_cell(text_val: str) -> Optional[str]:
    """
    Parse a Umanity date cell like '21Dec2025Han 11' → '2025-12-21'.
    Format: DDMonYYYYVenue RaceNo
    """
    if not text_val:
        return None
    text_val = text_val.strip().replace("\xa0", " ")
    m = re.match(r"(\d{1,2})([A-Za-z]{3})(\d{4})", text_val)
    if not m:
        return None
    day, mon_str, year = m.group(1), m.group(2), m.group(3)
    months = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
    }
    month = months.get(mon_str.capitalize())
    if not month:
        return None
    return f"{year}-{month}-{int(day):02d}"


def parse_race_weights(soup: BeautifulSoup) -> list[dict]:
    """
    Extract per-race horse body weights from the Umanity records table.

    Returns list of {"date": "2025-12-21", "weight": 478}.
    """
    records = []

    # Find the records table — it has column headers including 'HorseWt(Kg)'
    tables = soup.find_all("table")
    records_table = None
    for t in tables:
        header_row = t.find("tr")
        if header_row:
            header_text = header_row.get_text(strip=True)
            if "HorseWt(Kg)" in header_text and "DateVenue" in header_text:
                records_table = t
                break

    if not records_table:
        return records

    # Find the column index for HorseWt(Kg) and DateVenue/R
    rows = records_table.find_all("tr")
    # The header row has the column names
    header_cells = None
    for row in rows:
        cells = row.find_all(["th", "td"])
        texts = [c.get_text(strip=True) for c in cells]
        if "HorseWt(Kg)" in texts and "DateVenue/R" in texts:
            header_cells = texts
            break

    if not header_cells:
        return records

    try:
        date_idx = header_cells.index("DateVenue/R")
        weight_idx = header_cells.index("HorseWt(Kg)")
    except ValueError:
        return records

    # Parse data rows
    for row in rows:
        cells = row.find_all("td")
        if len(cells) <= max(date_idx, weight_idx):
            continue

        date_text = cells[date_idx].get_text(strip=True)
        weight_text = cells[weight_idx].get_text(strip=True)

        race_date = _parse_date_cell(date_text)
        weight = _parse_weight_cell(weight_text)

        if race_date and weight:
            records.append({"date": race_date, "weight": weight})

    return records


# ---------------------------------------------------------------------------
# DB Queries
# ---------------------------------------------------------------------------

def get_horses_missing_weight(limit: Optional[int] = None,
                               year: Optional[int] = None) -> list[dict]:
    """
    Get distinct horses that have at least one entry with NULL horse_weight.
    Only consider horses with valid netkeiba IDs (not synthetic JRA IDs).
    """
    query = """
        SELECT DISTINCT h.id, h.netkeiba_id, h.name_jp
        FROM horsebet.horses h
        JOIN horsebet.entries e ON e.horse_id = h.id
        JOIN horsebet.races r ON r.id = e.race_id
        WHERE e.horse_weight IS NULL
          AND h.netkeiba_id IS NOT NULL
          AND h.netkeiba_id NOT LIKE 'unknown_%%'
          AND h.netkeiba_id NOT LIKE 'jra_%%'
    """
    if year:
        query += f" AND EXTRACT(YEAR FROM r.date) = {year}"
    query += " ORDER BY h.id"
    if limit:
        query += f" LIMIT {limit}"

    with get_session() as session:
        rows = session.execute(text(query)).fetchall()
    return [{"id": r[0], "netkeiba_id": r[1], "name_jp": r[2]} for r in rows]


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _process_one_horse(horse: dict, idx: int, total: int,
                       dry_run: bool) -> dict:
    """Process a single horse. Returns stats dict."""
    result = {"updated": 0, "failed": 0, "skipped": 0}
    nk_id = horse["netkeiba_id"]
    url = HORSE_URL.format(code=nk_id)

    soup = _fetch(url)
    if not soup:
        log.debug(f"  [{idx}/{total}] ❌ Failed to fetch {horse['name_jp']} ({nk_id})")
        result["failed"] = 1
        return result

    race_weights = parse_race_weights(soup)
    if not race_weights:
        log.debug(f"  [{idx}/{total}] ⏭ No weight data for {horse['name_jp']} ({nk_id})")
        result["skipped"] = 1
        return result

    if dry_run:
        log.info(f"  [{idx}/{total}] 🔍 {horse['name_jp']}: {len(race_weights)} races with weight")
        for rw in race_weights[:3]:
            log.info(f"    {rw['date']} → {rw['weight']}kg")
        result["updated"] = len(race_weights)
        return result

    # Update DB: match entries by horse_id + race date
    with get_session() as session:
        for rw in race_weights:
            res = session.execute(text("""
                UPDATE horsebet.entries e
                SET horse_weight = :weight
                FROM horsebet.races r
                WHERE e.race_id = r.id
                  AND e.horse_id = :horse_id
                  AND r.date = :race_date
                  AND e.horse_weight IS NULL
            """), {
                "weight": rw["weight"],
                "horse_id": horse["id"],
                "race_date": rw["date"],
            })
            result["updated"] += res.rowcount

    if result["updated"] > 0:
        log.debug(
            f"  [{idx}/{total}] ✅ {horse['name_jp']}: "
            f"updated {result['updated']} entries"
        )

    return result


def backfill_weights(limit: Optional[int] = None, dry_run: bool = False,
                     workers: int = 4, year: Optional[int] = None):
    """Main backfill function with thread-pool concurrency."""
    horses = get_horses_missing_weight(limit=limit, year=year)
    total = len(horses)
    log.info(f"Found {total} horses with missing weights (workers={workers})")

    if not horses:
        log.info("Nothing to backfill!")
        return

    stats = {"updated": 0, "failed": 0, "skipped": 0, "horses_done": 0}
    lock = threading.Lock()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one_horse, horse, i, total, dry_run): i
            for i, horse in enumerate(horses, 1)
        }
        for future in as_completed(futures):
            r = future.result()
            with lock:
                stats["updated"] += r["updated"]
                stats["failed"] += r["failed"]
                stats["skipped"] += r["skipped"]
                stats["horses_done"] += 1
                done = stats["horses_done"]
                if done % 100 == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    log.info(
                        f"  [{done}/{total}] "
                        f"✅ {stats['updated']} entries | "
                        f"❌ {stats['failed']} | ⏭ {stats['skipped']} | "
                        f"{rate:.1f} horses/s | ETA {eta:.0f}s"
                    )

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 60}")
    log.info(
        f"Backfill complete in {elapsed:.0f}s: "
        f"{stats['updated']} entries updated, "
        f"{stats['failed']} failed, "
        f"{stats['skipped']} no data"
    )
    log.info(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UmaEdge — Backfill horse body weights from umanity.jp"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max horses to process")
    parser.add_argument("--year", type=int, default=None,
                        help="Only backfill entries from this year")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse only, don't update DB")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent workers (default: 4)")
    args = parser.parse_args()
    backfill_weights(
        limit=args.limit, dry_run=args.dry_run,
        workers=args.workers, year=args.year,
    )


if __name__ == "__main__":
    main()
