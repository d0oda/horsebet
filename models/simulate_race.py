"""
UmaEdge — Race Simulation Integration.

Wires the Monte Carlo PaceSimulator to real database data.
Given a race_id, loads entries and their historical features,
then runs the pace simulation to produce P(win) and P(top-3).

Usage:
    python -m models.simulate_race --race-id 1
"""

import argparse
import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from scraper.db import get_session
from models.pace_sim import PaceSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("simulate_race")


def load_race_entries(race_id: int) -> list[dict]:
    """
    Load entries for a race from the DB with the fields needed by PaceSimulator.
    Pulls historical stats inline to avoid full feature pipeline dependency.
    """
    query = """
        SELECT
            e.id AS entry_id,
            e.horse_id,
            h.name_jp AS horse_name,
            e.odds_win,
            e.post_position,
            r.field_size,
            r.distance,
            res.corner_positions,
            res.running_style
        FROM entries e
        JOIN races r ON r.id = e.race_id
        JOIN horses h ON h.id = e.horse_id
        LEFT JOIN results res ON res.entry_id = e.id
        WHERE e.race_id = :race_id
        ORDER BY e.post_position
    """

    with get_session() as session:
        rows = session.execute(text(query), {"race_id": race_id}).fetchall()

        entries = []
        for row in rows:
            horse_id = row[1]

            # Fetch historical stats for this horse
            hist = session.execute(text("""
                SELECT
                    res2.finish_pos,
                    res2.last_3f_secs,
                    res2.corner_positions
                FROM entries e2
                JOIN races r2 ON r2.id = e2.race_id
                LEFT JOIN results res2 ON res2.entry_id = e2.id
                WHERE e2.horse_id = :horse_id
                  AND res2.finish_pos IS NOT NULL
                ORDER BY r2.date DESC
                LIMIT 10
            """), {"horse_id": horse_id}).fetchall()

            # Compute rolling stats
            if hist:
                finishes = [h[0] for h in hist if h[0] is not None]
                last_3fs = [h[1] for h in hist if h[1] is not None]
                corners = []
                for h in hist:
                    if h[2]:
                        try:
                            corners.append(int(str(h[2]).split("-")[0].strip()))
                        except (ValueError, IndexError):
                            pass

                career_win_pct = sum(1 for f in finishes if f == 1) / len(finishes) if finishes else 0
                last3_avg_finish = np.mean(finishes[:3]) if finishes else None
                avg_last_3f = np.mean(last_3fs) if last_3fs else None
                best_last_3f = min(last_3fs) if last_3fs else None
                avg_first_corner = np.mean(corners) if corners else None
            else:
                career_win_pct = 0
                last3_avg_finish = None
                avg_last_3f = None
                best_last_3f = None
                avg_first_corner = None

            entries.append({
                "horse_id": horse_id,
                "horse_name": row[2],
                "entry_id": row[0],
                "odds_win": row[3],
                "post_position": row[4],
                "avg_first_corner": avg_first_corner,
                "avg_last_3f": avg_last_3f,
                "best_last_3f": best_last_3f,
                "career_win_pct": career_win_pct,
                "last3_avg_finish": last3_avg_finish,
                "running_style": row[8],
            })

    return entries


def simulate_race_by_id(race_id: int, n_simulations: int = 10000) -> dict:
    """
    Run pace simulation for a race from the database.

    Returns:
        Dict of {horse_id: {"win_prob": float, "place_prob": float, "style": str}}
    """
    entries = load_race_entries(race_id)
    if not entries:
        log.error(f"No entries found for race_id={race_id}")
        return {}

    # Get distance
    with get_session() as session:
        distance = session.execute(
            text("SELECT distance FROM races WHERE id = :rid"),
            {"rid": race_id}
        ).scalar() or 2000

    sim = PaceSimulator(n_simulations=n_simulations)
    results = sim.simulate_race(entries, distance=distance)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Pace Simulation for a Race")
    parser.add_argument("--race-id", type=int, required=True, help="Database race ID")
    parser.add_argument("--sims", type=int, default=10000, help="Number of simulations (default: 10000)")
    args = parser.parse_args()

    entries = load_race_entries(args.race_id)
    results = simulate_race_by_id(args.race_id, n_simulations=args.sims)

    if not results:
        print("No results — check that the race exists in the DB")
        return

    print(f"\n🏇 Pace Simulation — Race #{args.race_id} ({args.sims:,} iterations)")
    print("-" * 65)
    print(f"{'#':>3}  {'Horse':<16} {'Style':>4}  {'Win%':>6}  {'Place%':>7}  {'Odds':>5}")
    print("-" * 65)

    sorted_results = sorted(results.items(), key=lambda x: -x[1]["win_prob"])
    for hid, data in sorted_results:
        entry = next((e for e in entries if e["horse_id"] == hid), {})
        name = entry.get("horse_name", "?")[:14]
        odds = entry.get("odds_win", 0) or 0
        print(
            f"{entry.get('post_position', '?'):>3}  "
            f"{name:<16} {data['style']:>4}  "
            f"{data['win_prob']*100:>5.1f}%  "
            f"{data['place_prob']*100:>6.1f}%  "
            f"{odds:>5.1f}"
        )


if __name__ == "__main__":
    main()
