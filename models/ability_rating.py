"""
UmaEdge — Persistent Absolute Ability Rating (EWMA).

Computes a cross-race ability score (20–100 scale) for every horse,
updated after each race via exponential weighted moving average.

The score combines:
  base_speed_figure + class_adj + weight_adj + margin_adj + going_adj

Usage:
    from models.ability_rating import AbilityRatingEngine
    engine = AbilityRatingEngine()
    engine.backfill_all()          # one-time: process all historical results
    engine.update_for_date("2026-05-17")  # daily: after scraping results
"""

import json
import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from scraper.db import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ability_rating")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Class score mapping (higher = stronger race)
CLASS_SCORE = {
    "G1": 115, "GI": 115,
    "G2": 108, "GII": 108,
    "G3": 103, "GIII": 103,
    "L": 98, "OP": 94,
    "3勝": 90, "3win": 90,
    "2勝": 84, "2win": 84,
    "1勝": 78, "1win": 78,
    "未勝利": 70,
    "新馬": 65,
}

# EWMA decay factor: 70% old rating + 30% new race score
EWMA_ALPHA = 0.30

# Default rating for horses with no prior runs
DEFAULT_RATING = 55.0

# Rating bounds
RATING_MIN = 20.0
RATING_MAX = 150.0

# Margin text to beaten-lengths conversion
_MARGIN_TEXT = {
    "ハナ": 0.05, "クビ": 0.25, "アタマ": 0.1,
    "NS": 0.05, "NK": 0.25, "HD": 0.1,
    "大": 10.0, "DS": 10.0,
    "同着": 0.0, "DH": 0.0,
}


def _parse_margin(val) -> float:
    """Convert margin string to beaten lengths."""
    if pd.isna(val):
        return 0.0  # winner
    s = str(val).strip()
    if not s:
        return 0.0
    if s in _MARGIN_TEXT:
        return _MARGIN_TEXT[s]
    if '+' in s:
        s = s.split('+')[0].strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = re.match(r'^(\d+)/(\d+)$', s)
    if m:
        return int(m.group(1)) / int(m.group(2))
    m = re.match(r'^(\d+)[.\s](\d+)/(\d+)$', s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    return 0.0


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------

class AbilityRatingEngine:
    """Compute and persist EWMA ability ratings for horses."""

    def __init__(self):
        self._par_cache = {}  # (course_id, distance, going) -> (median_time, std_time)

    # ------------------------------------------------------------------
    # Race Score Computation
    # ------------------------------------------------------------------

    def compute_race_score(
        self,
        time_secs: float,
        distance: int,
        course_id: int,
        going: str,
        race_class: str,
        weight_carried: float,
        field_avg_weight: float,
        margin: str,
        finish_pos: int,
        field_size: int,
        par_time: Optional[float] = None,
        par_std: Optional[float] = None,
    ) -> dict:
        """
        Compute the absolute ability score for a single run.

        Returns dict with 'total' and component breakdown.
        """
        components = {}

        class_val = CLASS_SCORE.get(race_class, 78)

        # 1. Base speed figure
        if time_secs and time_secs > 0 and par_time and par_time > 0:
            if par_std and par_std > 0:
                base = class_val + ((par_time - time_secs) / max(par_std, 0.5)) * 10.0
            else:
                # Fallback: use percentage deviation × 1000
                base = class_val + ((par_time - time_secs) / par_time) * 1000.0
            components["base_speed"] = round(base, 2)
        else:
            # No timing data — approximate from class + position
            pos_penalty = max(0, (finish_pos - 1) * 2) if finish_pos else 10
            base = class_val - pos_penalty
            components["base_speed"] = round(base, 2)

        # 2. Class adjustment
        class_val = CLASS_SCORE.get(race_class, 78)
        class_adj = (class_val - 90) * 0.25
        components["class_adj"] = round(class_adj, 2)

        # 3. Weight adjustment
        if pd.notna(weight_carried) and pd.notna(field_avg_weight) and weight_carried > 0:
            weight_adj = (field_avg_weight - weight_carried) * -0.8
        else:
            weight_adj = 0.0
        components["weight_adj"] = round(weight_adj, 2)

        # 4. Margin adjustment
        beaten_lengths = _parse_margin(margin) if finish_pos != 1 else 0.0
        margin_adj = -min(beaten_lengths * 1.5, 12)
        components["margin_adj"] = round(margin_adj, 2)

        # 5. Going adjustment (simplified — forgive poor runs on unsuitable going)
        # This is a basic version; a more sophisticated one would use horse's
        # going preference from history. For now, no adjustment (we rely on
        # the condition_fit_score for going suitability).
        going_adj = 0.0
        components["going_adj"] = going_adj

        # Total
        total = base + class_adj + weight_adj + margin_adj + going_adj

        # Clamp to valid range
        total = max(RATING_MIN, min(RATING_MAX, total))
        components["total"] = round(total, 2)

        return components

    # ------------------------------------------------------------------
    # EWMA Update
    # ------------------------------------------------------------------

    @staticmethod
    def update_rating(old_rating: Optional[float], race_score: float) -> float:
        """EWMA update: new = old * (1-α) + score * α."""
        if old_rating is None:
            return race_score
        new = old_rating * (1 - EWMA_ALPHA) + race_score * EWMA_ALPHA
        return max(RATING_MIN, min(RATING_MAX, new))

    # ------------------------------------------------------------------
    # Par Time Lookup
    # ------------------------------------------------------------------

    def _load_par_times(self):
        """Load median and std of race times by (course, distance, going)."""
        if self._par_cache:
            return

        log.info("Loading par times from database...")
        query = """
            SELECT
                r.course_id,
                r.distance,
                r.going,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY res.time_secs) AS median_time,
                STDDEV(res.time_secs) AS std_time,
                COUNT(*) AS n
            FROM races r
            JOIN entries e ON e.race_id = r.id
            JOIN results res ON res.entry_id = e.id
            WHERE res.time_secs IS NOT NULL
              AND res.time_secs > 0
              AND r.course_id IS NOT NULL
              AND r.distance IS NOT NULL
              AND r.going IS NOT NULL
            GROUP BY r.course_id, r.distance, r.going
            HAVING COUNT(*) >= 500
        """
        with get_session() as session:
            rows = session.execute(text(query)).fetchall()

        for row in rows:
            key = (row[0], row[1], row[2])
            self._par_cache[key] = (float(row[3]), float(row[4]) if row[4] else None)

        log.info(f"Loaded {len(self._par_cache)} par time entries")

    def _get_par(self, course_id, distance, going):
        """Get (median_time, std_time) for a course/distance/going combo."""
        self._load_par_times()
        return self._par_cache.get((course_id, distance, going), (None, None))

    # ------------------------------------------------------------------
    # Backfill All Historical Ratings
    # ------------------------------------------------------------------

    def backfill_all(self, batch_size: int = 500):
        """
        Process ALL historical results chronologically and compute ratings.
        This is the one-time backfill job.
        """
        log.info("Starting full ability rating backfill...")
        self._load_par_times()

        # Load all results with needed data, sorted by date
        query = """
            SELECT
                r.id AS race_id,
                r.date,
                r.course_id,
                r.distance,
                r.going,
                r.class AS race_class,
                r.field_size,
                e.id AS entry_id,
                e.horse_id,
                e.weight_carried,
                res.finish_pos,
                res.margin,
                res.time_secs
            FROM races r
            JOIN entries e ON e.race_id = r.id
            JOIN results res ON res.entry_id = e.id
            WHERE res.finish_pos IS NOT NULL
            ORDER BY r.date, r.id, res.finish_pos
        """

        with get_session() as session:
            log.info("Loading all results...")
            rows = session.execute(text(query)).fetchall()
            columns = [
                "race_id", "date", "course_id", "distance", "going",
                "race_class", "field_size", "entry_id", "horse_id",
                "weight_carried", "finish_pos", "margin", "time_secs",
            ]

        df = pd.DataFrame(rows, columns=columns)
        log.info(f"Loaded {len(df)} results across {df['race_id'].nunique()} races")

        # Track current ratings in memory
        horse_ratings = {}  # horse_id -> current_rating
        history_rows = []

        # Pre-compute field average weights per race
        field_avg_weights = df.groupby("race_id")["weight_carried"].mean().to_dict()

        # Process race by race in chronological order
        race_groups = df.groupby("race_id", sort=False)
        total_races = df["race_id"].nunique()
        processed = 0

        for race_id, race_df in race_groups:
            field_avg_wt = field_avg_weights.get(race_id, 55.0)

            for _, row in race_df.iterrows():
                horse_id = row["horse_id"]
                par_time, par_std = self._get_par(
                    row["course_id"], row["distance"], row["going"]
                )

                components = self.compute_race_score(
                    time_secs=row["time_secs"],
                    distance=row["distance"],
                    course_id=row["course_id"],
                    going=row["going"] or "",
                    race_class=row["race_class"] or "",
                    weight_carried=row["weight_carried"],
                    field_avg_weight=field_avg_wt,
                    margin=row["margin"],
                    finish_pos=row["finish_pos"],
                    field_size=row["field_size"] or 16,
                    par_time=par_time,
                    par_std=par_std,
                )

                race_score = components["total"]
                old_rating = horse_ratings.get(horse_id)
                new_rating = self.update_rating(old_rating, race_score)
                horse_ratings[horse_id] = new_rating

                history_rows.append({
                    "horse_id": horse_id,
                    "race_id": race_id,
                    "race_date": str(row["date"]),
                    "rating_before": old_rating,
                    "race_score": race_score,
                    "rating_after": new_rating,
                    "components": json.dumps(
                        {k: (None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v)
                         for k, v in components.items()}
                    ),
                })

            processed += 1
            if processed % 1000 == 0:
                log.info(f"Processed {processed}/{total_races} races...")

        log.info(f"Computed ratings for {len(horse_ratings)} horses across {processed} races")

        # Write to database
        self._write_ratings(horse_ratings, history_rows, batch_size)

        # Summary stats
        ratings = list(horse_ratings.values())
        log.info(
            f"✅ Backfill complete. "
            f"Horses: {len(ratings)}, "
            f"Mean rating: {np.mean(ratings):.1f}, "
            f"Median: {np.median(ratings):.1f}, "
            f"Min: {np.min(ratings):.1f}, Max: {np.max(ratings):.1f}"
        )

    def _write_ratings(self, horse_ratings: dict, history_rows: list, batch_size: int):
        """Persist ratings to database using bulk inserts for performance."""
        import psycopg2.extras
        log.info(f"Writing {len(horse_ratings)} horse ratings and {len(history_rows)} history rows...")

        # Get raw psycopg2 connection for execute_values
        from scraper.db import engine
        raw_conn = engine.raw_connection()
        try:
            raw_conn.cursor().execute("SET search_path TO horsebet, public")
            cur = raw_conn.cursor()

            # Phase 1: Clear existing data
            cur.execute("DELETE FROM horse_rating_history")
            cur.execute("DELETE FROM horse_ratings")
            raw_conn.commit()
            log.info("Cleared existing rating history")

            # Phase 2: Bulk insert horse_ratings
            rating_data = [(hid, round(r, 2)) for hid, r in horse_ratings.items()]
            batch_sz = 500
            for i in range(0, len(rating_data), batch_sz):
                batch = rating_data[i:i + batch_sz]
                psycopg2.extras.execute_values(
                    cur,
                    "INSERT INTO horse_ratings (horse_id, ability_rating) VALUES %s "
                    "ON CONFLICT (horse_id) DO UPDATE SET ability_rating = EXCLUDED.ability_rating, updated_at = now()",
                    batch,
                )
                raw_conn.commit()
                if (i + batch_sz) % 5000 < batch_sz:
                    log.info(f"  horse_ratings: {min(i + batch_sz, len(rating_data))}/{len(rating_data)}")

            log.info(f"✅ Wrote {len(rating_data)} horse ratings")

            # Phase 3: Bulk update horses.ability_rating via temp table
            cur.execute("CREATE TEMP TABLE _tmp_ratings (horse_id INTEGER PRIMARY KEY, rating REAL)")
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO _tmp_ratings (horse_id, rating) VALUES %s",
                rating_data,
                page_size=5000,
            )
            cur.execute("""
                UPDATE horses h
                SET ability_rating = t.rating
                FROM _tmp_ratings t
                WHERE h.id = t.horse_id
            """)
            cur.execute("DROP TABLE _tmp_ratings")
            raw_conn.commit()
            log.info(f"✅ Updated {len(rating_data)} horses.ability_rating")

            # Phase 4: Bulk insert history
            history_data = [
                (r["horse_id"], r["race_id"], r["race_date"],
                 r["rating_before"], r["race_score"], r["rating_after"], r["components"])
                for r in history_rows
            ]
            for i in range(0, len(history_data), batch_sz):
                batch = history_data[i:i + batch_sz]
                psycopg2.extras.execute_values(
                    cur,
                    """INSERT INTO horse_rating_history
                        (horse_id, race_id, race_date, rating_before, race_score, rating_after, components)
                    VALUES %s""",
                    batch,
                    template="(%s, %s, %s, %s, %s, %s, %s::jsonb)",
                )
                raw_conn.commit()
                if (i + batch_sz) % 10000 < batch_sz:
                    log.info(f"  history: {min(i + batch_sz, len(history_data))}/{len(history_data)}")

            log.info("✅ Database write complete")
        finally:
            raw_conn.close()

    # ------------------------------------------------------------------
    # Daily Update (after scraping results)
    # ------------------------------------------------------------------

    def update_for_date(self, race_date: str):
        """
        Update ratings for all horses that raced on a given date.
        Call this after scraping results for a race day.
        """
        self._load_par_times()

        query = """
            SELECT
                r.id AS race_id,
                r.course_id, r.distance, r.going,
                r.class AS race_class, r.field_size,
                e.horse_id, e.weight_carried,
                res.finish_pos, res.margin, res.time_secs,
                COALESCE(hr.ability_rating, h.ability_rating) AS current_rating
            FROM races r
            JOIN entries e ON e.race_id = r.id
            JOIN results res ON res.entry_id = e.id
            JOIN horses h ON h.id = e.horse_id
            LEFT JOIN horse_ratings hr ON hr.horse_id = e.horse_id
            WHERE r.date = :race_date
              AND res.finish_pos IS NOT NULL
            ORDER BY r.id, res.finish_pos
        """

        with get_session() as session:
            rows = session.execute(text(query), {"race_date": race_date}).fetchall()
            columns = [
                "race_id", "course_id", "distance", "going",
                "race_class", "field_size", "horse_id", "weight_carried",
                "finish_pos", "margin", "time_secs", "current_rating",
            ]

        if not rows:
            log.info(f"No results found for {race_date}")
            return

        df = pd.DataFrame(rows, columns=columns)
        field_avg_weights = df.groupby("race_id")["weight_carried"].mean().to_dict()

        updated = 0
        with get_session() as session:
            for _, row in df.iterrows():
                par_time, par_std = self._get_par(row["course_id"], row["distance"], row["going"])
                field_avg_wt = field_avg_weights.get(row["race_id"], 55.0)

                components = self.compute_race_score(
                    time_secs=row["time_secs"],
                    distance=row["distance"],
                    course_id=row["course_id"],
                    going=row["going"] or "",
                    race_class=row["race_class"] or "",
                    weight_carried=row["weight_carried"],
                    field_avg_weight=field_avg_wt,
                    margin=row["margin"],
                    finish_pos=row["finish_pos"],
                    field_size=row["field_size"] or 16,
                    par_time=par_time,
                    par_std=par_std,
                )

                race_score = components["total"]
                old_rating = row["current_rating"]
                new_rating = self.update_rating(old_rating, race_score)

                # Update horse rating (both tables)
                session.execute(
                    text("UPDATE horses SET ability_rating = :rating WHERE id = :hid"),
                    {"rating": round(new_rating, 2), "hid": row["horse_id"]},
                )
                session.execute(
                    text("""INSERT INTO horse_ratings (horse_id, ability_rating)
                            VALUES (:hid, :rating)
                            ON CONFLICT (horse_id) DO UPDATE SET
                                ability_rating = EXCLUDED.ability_rating,
                                updated_at = now()"""),
                    {"hid": row["horse_id"], "rating": round(new_rating, 2)},
                )

                # Insert history (upsert)
                session.execute(
                    text("""
                        INSERT INTO horse_rating_history
                            (horse_id, race_id, race_date, rating_before, race_score, rating_after, components)
                        VALUES
                            (:horse_id, :race_id, :race_date, :rating_before, :race_score, :rating_after, CAST(:components AS jsonb))
                        ON CONFLICT (horse_id, race_id) DO UPDATE SET
                            rating_before = EXCLUDED.rating_before,
                            race_score = EXCLUDED.race_score,
                            rating_after = EXCLUDED.rating_after,
                            components = EXCLUDED.components
                    """),
                    {
                        "horse_id": row["horse_id"],
                        "race_id": row["race_id"],
                        "race_date": race_date,
                        "rating_before": old_rating,
                        "race_score": race_score,
                        "rating_after": new_rating,
                        "components": json.dumps(components),
                    },
                )
                updated += 1

        log.info(f"✅ Updated {updated} ratings for {race_date}")

    # ------------------------------------------------------------------
    # Lookup helpers (for feature engineering)
    # ------------------------------------------------------------------

    @staticmethod
    def get_horse_rating(horse_id: int) -> Optional[float]:
        """Get current ability rating for a horse."""
        with get_session() as session:
            # Try horse_ratings cache first, fall back to horses table
            result = session.execute(
                text("SELECT ability_rating FROM horse_ratings WHERE horse_id = :hid"),
                {"hid": horse_id},
            ).scalar()
            if result is None:
                result = session.execute(
                    text("SELECT ability_rating FROM horses WHERE id = :hid"),
                    {"hid": horse_id},
                ).scalar()
        return result

    @staticmethod
    def get_rating_history(horse_id: int, before_date: str = None) -> pd.DataFrame:
        """Get rating history for a horse, optionally filtered before a date."""
        where = "WHERE horse_id = :hid"
        params = {"hid": horse_id}
        if before_date:
            where += " AND race_date < :before"
            params["before"] = before_date

        query = f"""
            SELECT race_date, race_score, rating_after
            FROM horse_rating_history
            {where}
            ORDER BY race_date
        """
        with get_session() as session:
            rows = session.execute(text(query), params).fetchall()
        return pd.DataFrame(rows, columns=["race_date", "race_score", "rating_after"])

    @staticmethod
    def get_field_ratings(race_id: int) -> dict:
        """Get ability ratings for all horses in a race."""
        query = """
            SELECT e.horse_id, COALESCE(hr.ability_rating, h.ability_rating, :default) AS rating
            FROM entries e
            JOIN horses h ON h.id = e.horse_id
            LEFT JOIN horse_ratings hr ON hr.horse_id = e.horse_id
            WHERE e.race_id = :rid
        """
        with get_session() as session:
            rows = session.execute(
                text(query), {"rid": race_id, "default": DEFAULT_RATING}
            ).fetchall()
        return {row[0]: row[1] for row in rows}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UmaEdge — Ability Rating Engine")
    parser.add_argument("--backfill", action="store_true", help="Full historical backfill")
    parser.add_argument("--date", type=str, help="Update ratings for a specific date (YYYY-MM-DD)")
    parser.add_argument("--horse", type=int, help="Show rating for a specific horse ID")
    args = parser.parse_args()

    engine = AbilityRatingEngine()

    if args.backfill:
        engine.backfill_all()
    elif args.date:
        engine.update_for_date(args.date)
    elif args.horse:
        rating = engine.get_horse_rating(args.horse)
        history = engine.get_rating_history(args.horse)
        print(f"\nHorse {args.horse} — Current Rating: {rating}")
        if not history.empty:
            print(f"Rating History ({len(history)} runs):")
            for _, row in history.iterrows():
                print(f"  {row['race_date']}: score={row['race_score']:.1f} → rating={row['rating_after']:.1f}")
    else:
        parser.print_help()
