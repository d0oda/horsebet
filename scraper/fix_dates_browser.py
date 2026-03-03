"""
UmaEdge — Fix Race Dates via Browser Scraping.

Uses Playwright (headless browser) to scrape actual race dates from
en.netkeiba.com, bypassing the anti-bot HTTP 403 blocks.

Strategy:
    1. Get all unique (year, course, meeting, round) combos with bad dates
    2. For each combo, fetch ONE race page from en.netkeiba.com
    3. Extract the actual date from the page
    4. Batch-update all 12 races (NN=01-12) sharing that combo

This is ~438 browser page loads instead of 5,245 HTTP requests.

Usage:
    python -m scraper.fix_dates_browser              # fix all
    python -m scraper.fix_dates_browser --dry-run     # preview only
    python -m scraper.fix_dates_browser --year 2025   # fix specific year
"""

import argparse
import asyncio
import logging
import os
import re
import sys

from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scraper.db import get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fix_dates_browser")

# en.netkeiba.com works from browser (jp subdomain blocked)
EN_RESULT_URL = "https://en.netkeiba.com/race/result.html?race_id={race_id}"


def get_bad_date_combos(year_filter: int = None) -> list[dict]:
    """Get unique (year, course, meeting, round) combos with bad dates."""
    with get_session() as session:
        query = """
            SELECT 
                SUBSTRING(netkeiba_id FROM 1 FOR 10) as combo_prefix,
                COUNT(*) as race_count,
                MIN(netkeiba_id) as sample_id,
                date::text as bad_date
            FROM horsebet.races
            WHERE date IN (
                SELECT date FROM horsebet.races
                GROUP BY date HAVING COUNT(*) > 36
            )
        """
        if year_filter:
            query += f" AND netkeiba_id LIKE '{year_filter}%'"
        query += """
            GROUP BY combo_prefix, date
            ORDER BY combo_prefix
        """
        rows = session.execute(text(query)).fetchall()

    return [
        {
            "prefix": r[0],
            "count": r[1],
            "sample_id": r[2],
            "bad_date": r[3],
        }
        for r in rows
    ]


async def fetch_race_date(page, race_id: str) -> str | None:
    """Fetch actual date from en.netkeiba.com race page using Playwright."""
    url = EN_RESULT_URL.format(race_id=race_id)
    try:
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        if resp and resp.status != 200:
            # Try Japanese result page as fallback
            jp_url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
            resp = await page.goto(jp_url, wait_until="domcontentloaded", timeout=20000)

        content = await page.content()

        # en.netkeiba uses "2025/1/5(Sun)" format
        m = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", content)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

        # Japanese format: 2025年1月5日
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", content)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    except Exception as e:
        log.warning(f"  Error fetching {race_id}: {e}")
    return None


def update_races_for_combo(prefix: str, new_date: str, dry_run: bool = False):
    """Update all races matching the combo prefix to the new date."""
    if dry_run:
        return

    with get_session() as session:
        session.execute(
            text("""
                UPDATE horsebet.races 
                SET date = :new_date 
                WHERE SUBSTRING(netkeiba_id FROM 1 FOR 10) = :prefix
            """),
            {"new_date": new_date, "prefix": prefix},
        )


async def run_fix(combos: list[dict], dry_run: bool = False):
    """Run the date fix using Playwright browser."""
    from playwright.async_api import async_playwright

    total = len(combos)
    fixed = 0
    failed = 0
    already_correct = 0

    log.info(f"Starting browser-based date fix for {total} combos...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = await context.new_page()

        for i, combo in enumerate(combos, 1):
            sample_id = combo["sample_id"]
            actual_date = await fetch_race_date(page, sample_id)

            if not actual_date:
                failed += 1
                if i <= 20 or i % 50 == 0:
                    log.warning(f"  [{i}/{total}] ❌ Failed: {sample_id}")
                continue

            if actual_date == combo["bad_date"]:
                already_correct += 1
                continue

            fixed += 1
            if i <= 50 or i % 50 == 0:
                log.info(
                    f"  [{i}/{total}] {combo['prefix']}: "
                    f"{combo['bad_date']} → {actual_date} "
                    f"({combo['count']} races)"
                )

            update_races_for_combo(combo["prefix"], actual_date, dry_run)

            # Brief pause every 20 requests
            if i % 20 == 0:
                await asyncio.sleep(1)

            # Progress log
            if i % 100 == 0:
                log.info(
                    f"  Progress: {i}/{total} "
                    f"(fixed: {fixed} | failed: {failed} | correct: {already_correct})"
                )

        await browser.close()

    log.info(f"\n{'=' * 60}")
    log.info(f"  Date Fix Summary")
    log.info(f"{'=' * 60}")
    log.info(f"  Total combos:    {total}")
    log.info(f"  Fixed:           {fixed}")
    log.info(f"  Failed:          {failed}")
    log.info(f"  Already correct: {already_correct}")
    if dry_run:
        log.info(f"  (dry run — no changes made)")
    log.info(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Fix Race Dates (Browser)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--year", type=int, help="Fix only specific year")
    args = parser.parse_args()

    combos = get_bad_date_combos(args.year)
    log.info(f"Found {len(combos)} unique combos with bad dates")

    if not combos:
        log.info("Nothing to fix!")
        return

    asyncio.run(run_fix(combos, args.dry_run))


if __name__ == "__main__":
    main()
