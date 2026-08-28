#!/usr/bin/env python3
"""
UmaEdge — Database Health Monitor.

A daily diagnostic script that verifies data parity and flags missing
critical fields (finish_pos, sectional times, horse weights) across
recently scraped races.

If critical data is missing, this script exits with status 1, which
can fail a CI/CD pipeline or trigger an alert.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from scraper.db import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("db_health")

def check_missing_results():
    """Check for past races (older than 1 day) completely missing finish positions."""
    with get_session() as session:
        query = text("""
            SELECT r.date, r.netkeiba_id, COUNT(e.id) as total_entries
            FROM races r
            JOIN entries e ON r.id = e.race_id
            LEFT JOIN results res ON e.id = res.entry_id
            WHERE r.date >= date('now', '-30 days')
              AND r.date < date('now', '-1 day')
            GROUP BY r.id, r.date, r.netkeiba_id
            HAVING COUNT(res.finish_pos) = 0 AND COUNT(e.id) > 0
            ORDER BY r.date DESC;
        """)
        rows = session.execute(query).fetchall()
        
    if rows:
        log.error(f"Found {len(rows)} past races missing finish positions entirely!")
        for r in rows[:5]:
            log.error(f"  - Race {r.netkeiba_id} on {r.date}: {r.total_entries} entries with 0 results")
        return len(rows)
    return 0

def check_results_entries_parity():
    """Check for parity between results table and entries table finish_pos."""
    with get_session() as session:
        query = text("""
            SELECT r.date, r.netkeiba_id, COUNT(e.id) as desynced_entries
            FROM races r
            JOIN entries e ON r.id = e.race_id
            JOIN results res ON e.id = res.entry_id
            WHERE e.finish_pos IS NULL AND res.finish_pos IS NOT NULL
            GROUP BY r.id, r.date, r.netkeiba_id
            ORDER BY r.date DESC;
        """)
        rows = session.execute(query).fetchall()
        
    if rows:
        log.error(f"Found {len(rows)} races with results desynchronized from entries table!")
        for r in rows[:5]:
            log.error(f"  - Race {r.netkeiba_id} on {r.date}: {r.desynced_entries} entries missing finish_pos in entries")
        return len(rows)
    return 0

def check_missing_sectionals():
    """Check for missing time_secs or last_3f_secs in recent results."""
    with get_session() as session:
        query = text("""
            SELECT r.date, r.netkeiba_id, COUNT(e.id) as missing_sectionals
            FROM races r
            JOIN entries e ON r.id = e.race_id
            LEFT JOIN results res ON e.id = res.entry_id
            WHERE r.date >= date('now', '-30 days')
              AND res.finish_pos IS NOT NULL
              AND (res.time_secs IS NULL OR res.last_3f_secs IS NULL)
            GROUP BY r.id, r.date, r.netkeiba_id
            HAVING COUNT(e.id) > 0
            ORDER BY r.date DESC;
        """)
        rows = session.execute(query).fetchall()
        
    if rows:
        log.error(f"Found {len(rows)} recent races missing sectional times!")
        for r in rows[:5]:
            log.error(f"  - Race {r.netkeiba_id} on {r.date}: {r.missing_sectionals} entries missing sectionals")
        return len(rows)
    return 0

def check_missing_weights():
    """Check for missing horse weights in recent races."""
    with get_session() as session:
        query = text("""
            SELECT r.date, r.netkeiba_id, COUNT(e.id) as missing_weights
            FROM races r
            JOIN entries e ON r.id = e.race_id
            WHERE r.date >= date('now', '-30 days')
              AND e.horse_weight IS NULL
            GROUP BY r.id, r.date, r.netkeiba_id
            HAVING COUNT(e.id) > 0
            ORDER BY r.date DESC;
        """)
        rows = session.execute(query).fetchall()
        
    if rows:
        log.warning(f"Found {len(rows)} recent races missing horse body weights!")
        for r in rows[:5]:
            log.warning(f"  - Race {r.netkeiba_id} on {r.date}: {r.missing_weights} entries missing weights")
        return len(rows)
    return 0

def main():
    log.info("Starting database health verification...")
    errors = 0
    
    errors += check_missing_results()
    errors += check_results_entries_parity()
    errors += check_missing_sectionals()
    
    # Body weight gaps are warnings, not immediate failures for predictions
    check_missing_weights()
    
    if errors > 0:
        log.error(f"Health check failed with {errors} critical errors.")
        sys.exit(1)
        
    log.info("✅ Database health is excellent. All checks passed.")
    sys.exit(0)

if __name__ == "__main__":
    main()
