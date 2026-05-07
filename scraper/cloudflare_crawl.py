"""
UmaEdge — Cloudflare Crawl Scraper.

Scrapes historical race data from netkeiba using the Cloudflare Browser Rendering Crawl API.
Great for backfilling entire missing years at scale without getting rate limited or
spending days running sequential requests.

Usage:
    # Scrape missing races for a year
    python -m scraper.cloudflare_crawl --year 2024

    # Dry run - test configuration on one page
    python -m scraper.cloudflare_crawl --year 2024 --test
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from scraper.db import get_session
from scraper.netkeiba import parse_race_page

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cloudflare_crawl")

# Configuration
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

CRAWL_API_URL = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/browser-rendering/crawl"
HEADERS = {
    "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
    "Content-Type": "application/json"
}


def _verify_credentials():
    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN:
        log.error("Missing Cloudflare API credentials. Please set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env")
        sys.exit(1)


def start_crawl_job(year: int, is_test: bool = False) -> str:
    """Submit a request to start a crawl job for a given year."""
    log.info(f"Starting crawl job for {year} {'(Test Mode)' if is_test else ''}...")
    
    # We start crawling from the top-level calendar page for the year
    start_url = f"https://race.netkeiba.com/top/calendar.html?year={year}"
    
    # In test mode, we just want to hit the calendar page and maybe one race date page
    # In full mode, we want to discover all race ids for the year
    payload = {
        "url": start_url,
        "render": False,  # Save time: we largely just need static HTML for parsing
    }
    
    # We only want to process actual race result pages:
    # Example valid target: https://race.netkeiba.com/race/result.html?race_id=202405010101
    
    # The Cloudflare Browser Rendering Crawl API doesn't currently support 'include' globs
    # on the root payload.
    if not is_test:
        pass
    else:
        # In test mode, bypass the top-level calendar and start directly on a race result 
        # so we guarantee we parse at least one valid page.
        start_url = "https://race.netkeiba.com/race/result.html?race_id=202405010101"
        payload["url"] = start_url
        payload["maxOptions"] = {"maxPages": 2}
    
    response = requests.post(CRAWL_API_URL, headers=HEADERS, json=payload)
    
    if response.status_code != 200:
        log.error(f"Failed to start crawl job: {response.text}")
        sys.exit(1)
        
    data = response.json()
    if not data.get("success"):
        log.error(f"Crawl job failed: {data}")
        sys.exit(1)
        
    job_id = data.get("result")
    log.info(f"✅ Job initiated. Job ID: {job_id}")
    return job_id


def poll_job_status(job_id: str) -> dict:
    """Poll the Cloudflare API until the crawl job is finished."""
    url = f"{CRAWL_API_URL}/{job_id}"
    
    while True:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            log.error(f"Failed to fetch job status: {response.status_code} - {response.text}")
            time.sleep(10)
            continue
            
        data = response.json()
        if not data.get("success"):
             log.error(f"Could not fetch job data: {data}")
             continue
             
        status = data.get("result", {}).get("status")
        log.info(f"Job Status: {status}")
        
        if status in ["completed", "cancelled_due_to_timeout", "cancelled_due_to_limits", "cancelled_by_user", "errored"]:
            return data.get("result", {})
            
        # Still running. Crawl jobs can take a while.
        time.sleep(15)


def process_crawled_results(job_data: dict, dry_run: bool = False):
    """Process the results from the completed crawl job."""
    from scraper.netkeiba import parse_race_page, save_race_to_db
    from bs4 import BeautifulSoup
    
    log.info("\nProcessing results...")
    
    records = job_data.get("records", [])
    log.info(f"Crawler returned {len(records)} records. Parsing HTML payloads...")
    
    inserted = 0
    failed = 0
    
    for idx, record in enumerate(records):
        url = record.get("url", "")
        status = record.get("status")
        html = record.get("html")
        
        if not url or "result.html" not in url:
            continue
            
        if status != "completed" or not html:
            log.warning(f"  [{idx}] ⏭ Skipping {url} - bad status or no HTML")
            failed += 1
            continue
            
        # Extract netkeiba_id from the URL (e.g. ?race_id=202405010101)
        import re
        m = re.search(r"race_id=(\d+)", url)
        if not m:
            continue
            
        nk_id = m.group(1)
        
        try:
            # Parse the EUC-JP mangled HTML using our existing parser
            soup = BeautifulSoup(html, "lxml")
            race_data = parse_race_page(soup, nk_id)
            
            if not race_data:
                log.warning(f"  [{idx}] ❌ Parse failed for {nk_id}")
                failed += 1
                continue
                
            if dry_run:
                log.info(f"  [{idx}] ✅ {race_data.race_name_jp} ({len(race_data.entries)} entries) (Dry Run)")
                inserted += 1
                continue
                
            # Use the existing DB ingestion logic
            save_race_to_db(race_data)
                
            log.info(f"  [{idx}] ✅ Inserted {race_data.race_name_jp} ({len(race_data.entries)} entries)")
            inserted += 1
            
        except Exception as e:
            log.error(f"  [{idx}] ❌ Crash on {nk_id}: {e}")
            failed += 1
            
    log.info(f"\n✅ Completed chunk: {inserted} inserted, {failed} failed.")

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Cloudflare Crawl Scraper")
    parser.add_argument("--year", type=int, required=True, help="Year to scrape (e.g. 2024)")
    parser.add_argument("--test", action="store_true", help="Test mode: minimal crawl scope")
    args = parser.parse_args()

    _verify_credentials()
    
    job_id = start_crawl_job(args.year, args.test)
    job_data = poll_job_status(job_id)
    
    if job_data.get("status") == "completed":
        process_crawled_results(job_data, args.test)
    else:
        log.warning(f"Job finished with final status: {job_data.get('status')}")


if __name__ == "__main__":
    main()
