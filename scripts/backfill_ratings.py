#!/usr/bin/env python
"""
UmaEdge — Backfill Ability Ratings.

One-time script to compute EWMA ability ratings for all historical results.
Must be run after migrations/006_ability_rating.sql has been applied.

Usage:
    python scripts/backfill_ratings.py
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ability_rating import AbilityRatingEngine


def main():
    print("=" * 60)
    print("  UmaEdge — Ability Rating Backfill")
    print("=" * 60)
    print()
    print("This will process all historical results chronologically")
    print("and compute EWMA ability ratings for every horse.")
    print()

    engine = AbilityRatingEngine()
    engine.backfill_all()

    print()
    print("Done! You can now verify ratings with:")
    print("  python -m models.ability_rating --horse <horse_id>")


if __name__ == "__main__":
    main()
