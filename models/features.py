"""
UmaEdge — Feature Engineering Pipeline.

Transforms raw race/horse/jockey data from the horsebet schema into
model-ready feature vectors. Each feature vector represents one entry
(horse-in-race) and contains ~40-60 numerical features.

Usage:
    from models.features import FeatureBuilder
    
    fb = FeatureBuilder()
    df = fb.build_features_for_race(race_id=123)
    # or
    df = fb.build_features_all()  # all historical races
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from scraper.db import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("features")

# ---------------------------------------------------------------------------
# Class hierarchy mapping (lower = stronger)
# ---------------------------------------------------------------------------

CLASS_RANK = {
    "G1": 1, "GI": 1,
    "G2": 2, "GII": 2,
    "G3": 3, "GIII": 3,
    "L": 4, "OP": 5,
    "3勝": 6, "3win": 6,
    "2勝": 7, "2win": 7,
    "1勝": 8, "1win": 8,
    "未勝利": 9,
    "新馬": 10,
}

GOING_MAP = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
SURFACE_MAP = {"turf": 0, "dirt": 1}
SEX_MAP = {"牡": 0, "牝": 1, "セ": 2}  # male, female, gelding

# All odds-derived features (raw + z-score normalised).
# Excluding these forces the model to learn fundamental signals only.
ODDS_FEATURES = [
    "odds_win", "log_odds", "popularity",
    "odds_win_z", "log_odds_z", "popularity_z",
]


class FeatureBuilder:
    """Build feature vectors for race entries from the database."""

    def __init__(self):
        self._horse_history_cache = {}
        self._pace_cache = {}  # race_id -> {horse_id: {win_prob, place_prob, style}}

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def _load_race_data(self, race_id: Optional[int] = None) -> pd.DataFrame:
        """Load joined race/entry/result data. If race_id=None, load all."""
        where = "AND r.id = :race_id" if race_id else ""
        query = f"""
            SELECT
                r.id AS race_id,
                r.netkeiba_id AS race_nk_id,
                r.date,
                r.course_id,
                r.race_number,
                r.distance,
                r.surface,
                r.going,
                r.class AS race_class,
                r.grade,
                r.weather,
                r.field_size,
                e.id AS entry_id,
                e.horse_id,
                e.jockey_id,
                e.draw,
                e.post_position,
                e.weight_carried,
                e.horse_weight,
                e.horse_weight_change,
                e.odds_win,
                e.popularity,
                h.name_jp AS horse_name,
                h.sex,
                h.birth_year,
                h.netkeiba_id AS horse_nk_id,
                h.trainer_id,
                j.name_jp AS jockey_name,
                res.finish_pos,
                res.margin,
                res.time_secs,
                res.last_3f_secs,
                res.corner_positions,
                res.running_style
            FROM races r
            JOIN entries e ON e.race_id = r.id
            JOIN horses h ON h.id = e.horse_id
            LEFT JOIN jockeys j ON j.id = e.jockey_id
            LEFT JOIN results res ON res.entry_id = e.id
            WHERE r.field_size IS NOT NULL
            {where}
            ORDER BY r.date, r.id, e.post_position
        """

        with get_session() as session:
            params = {"race_id": race_id} if race_id else {}
            result = session.execute(text(query), params)
            rows = result.fetchall()
            columns = result.keys()

        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            log.warning("No race data loaded")
        else:
            # Ensure date is string for consistent comparison
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
            log.info(f"Loaded {len(df)} entries across {df['race_id'].nunique()} races")
        return df

    def _load_horse_history(self) -> pd.DataFrame:
        """Load all historical results per horse for rolling calculations."""
        query = """
            SELECT
                e.horse_id,
                e.jockey_id,
                r.date,
                r.distance,
                r.surface,
                r.going,
                r.class AS race_class,
                r.grade,
                r.course_id,
                e.draw,
                e.weight_carried,
                e.horse_weight,
                e.odds_win,
                res.finish_pos,
                res.time_secs,
                res.last_3f_secs,
                res.corner_positions,
                r.field_size,
                h.trainer_id
            FROM entries e
            JOIN races r ON r.id = e.race_id
            JOIN horses h ON h.id = e.horse_id
            LEFT JOIN results res ON res.entry_id = e.id
            WHERE res.finish_pos IS NOT NULL
            ORDER BY e.horse_id, r.date
        """
        with get_session() as session:
            result = session.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()

        df = pd.DataFrame(rows, columns=columns)
        # Ensure date is string for consistent comparison
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)
        return df

    # ------------------------------------------------------------------
    # Feature Computation — Per Horse Rolling History
    # ------------------------------------------------------------------

    def _horse_rolling_features(
        self, horse_id: int, race_date: str, distance: int,
        surface: str, course_id: Optional[int], history_df: pd.DataFrame
    ) -> dict:
        """Compute rolling features for a horse based on their past races."""
        # Filter to this horse's history BEFORE the current race
        hist = history_df[
            (history_df["horse_id"] == horse_id)
            & (history_df["date"] < race_date)
        ].sort_values("date", ascending=False)

        features = {}

        if hist.empty:
            # First-time runner — return NaN features
            return self._empty_horse_features()

        # --- Recent form (last N races) ---
        for n in [3, 5, 10]:
            recent = hist.head(n)
            prefix = f"last{n}"

            features[f"{prefix}_win_pct"] = (recent["finish_pos"] == 1).mean() if len(recent) > 0 else np.nan
            features[f"{prefix}_place_pct"] = (recent["finish_pos"] <= 3).mean() if len(recent) > 0 else np.nan
            features[f"{prefix}_avg_finish"] = recent["finish_pos"].mean() if len(recent) > 0 else np.nan
            features[f"{prefix}_avg_beaten_pct"] = (
                (recent["finish_pos"] / recent["field_size"]).mean()
                if len(recent) > 0 and (recent["field_size"] > 0).all()
                else np.nan
            )

        # --- Speed features ---
        features["best_time_at_dist"] = (
            hist[hist["distance"] == distance]["time_secs"].min()
            if len(hist[hist["distance"] == distance]) > 0
            else np.nan
        )
        features["avg_last_3f"] = hist["last_3f_secs"].mean()
        features["best_last_3f"] = hist["last_3f_secs"].min()

        # Last race speed figure (normalised time)
        last_race = hist.iloc[0]
        features["last_time_secs"] = last_race["time_secs"]
        features["last_last_3f"] = last_race["last_3f_secs"]
        features["last_finish_pos"] = last_race["finish_pos"]

        # --- Freshness ---
        try:
            last_date = pd.to_datetime(last_race["date"])
            current_date = pd.to_datetime(race_date)
            features["days_since_last"] = (current_date - last_date).days
        except Exception:
            features["days_since_last"] = np.nan

        # --- Win streak / losing streak ---
        streak = 0
        for _, row in hist.iterrows():
            if row["finish_pos"] == 1:
                streak += 1
            else:
                break
        features["win_streak"] = streak

        # --- Distance affinity ---
        dist_hist = hist[hist["distance"] == distance]
        features["runs_at_dist"] = len(dist_hist)
        features["win_pct_at_dist"] = (
            (dist_hist["finish_pos"] == 1).mean() if len(dist_hist) > 0 else np.nan
        )

        # --- Surface affinity ---
        surf_hist = hist[hist["surface"] == surface]
        features["runs_on_surface"] = len(surf_hist)
        features["win_pct_on_surface"] = (
            (surf_hist["finish_pos"] == 1).mean() if len(surf_hist) > 0 else np.nan
        )

        # --- Course affinity ---
        if course_id is not None:
            course_hist = hist[hist["course_id"] == course_id]
            features["runs_at_course"] = len(course_hist)
            features["win_pct_at_course"] = (
                (course_hist["finish_pos"] == 1).mean() if len(course_hist) > 0 else np.nan
            )
        else:
            features["runs_at_course"] = 0
            features["win_pct_at_course"] = np.nan

        # --- Class level ---
        features["best_class_rank"] = (
            hist["race_class"].map(CLASS_RANK).min()
            if hist["race_class"].map(CLASS_RANK).notna().any()
            else 10
        )

        # --- Career stats ---
        features["career_runs"] = len(hist)
        features["career_wins"] = (hist["finish_pos"] == 1).sum()
        features["career_win_pct"] = features["career_wins"] / features["career_runs"]
        features["career_place_pct"] = (hist["finish_pos"] <= 3).sum() / features["career_runs"]

        # --- Early speed proxy (average first corner position) ---
        corners = hist["corner_positions"].dropna()
        first_corners = []
        for c in corners:
            try:
                first = int(str(c).split("-")[0].strip())
                first_corners.append(first)
            except (ValueError, IndexError):
                continue
        features["avg_first_corner"] = np.mean(first_corners) if first_corners else np.nan

        # --- Weight trend ---
        weights = hist["horse_weight"].dropna()
        if len(weights) >= 2:
            features["weight_trend"] = weights.iloc[0] - weights.iloc[1]
        else:
            features["weight_trend"] = 0

        return features

    def _empty_horse_features(self) -> dict:
        """Return NaN features for first-time runners."""
        keys = [
            "last3_win_pct", "last3_place_pct", "last3_avg_finish", "last3_avg_beaten_pct",
            "last5_win_pct", "last5_place_pct", "last5_avg_finish", "last5_avg_beaten_pct",
            "last10_win_pct", "last10_place_pct", "last10_avg_finish", "last10_avg_beaten_pct",
            "best_time_at_dist", "avg_last_3f", "best_last_3f",
            "last_time_secs", "last_last_3f", "last_finish_pos",
            "days_since_last", "win_streak",
            "runs_at_dist", "win_pct_at_dist",
            "runs_on_surface", "win_pct_on_surface",
            "runs_at_course", "win_pct_at_course",
            "best_class_rank", "career_runs", "career_wins",
            "career_win_pct", "career_place_pct",
            "avg_first_corner", "weight_trend",
        ]
        return {k: np.nan for k in keys}

    # ------------------------------------------------------------------
    # Feature Computation — Jockey Rolling History
    # ------------------------------------------------------------------

    def _jockey_features(self, jockey_id: Optional[int], race_date: str, history_df: pd.DataFrame) -> dict:
        """Compute rolling jockey features with 5/10 windows and ROI."""
        null_feats = {
            "jockey_win_pct_5": np.nan,
            "jockey_win_pct_10": np.nan,
            "jockey_place_pct_5": np.nan,
            "jockey_place_pct_10": np.nan,
            "jockey_roi_10": np.nan,
            "jockey_recent_wins": np.nan,
        }

        if jockey_id is None:
            return null_feats

        hist = history_df[
            (history_df["jockey_id"] == jockey_id)
            & (history_df["date"] < race_date)
        ].sort_values("date", ascending=False)

        if hist.empty:
            return null_feats

        features = {}

        for n in [5, 10]:
            recent = hist.head(n)
            features[f"jockey_win_pct_{n}"] = (recent["finish_pos"] == 1).mean()
            features[f"jockey_place_pct_{n}"] = (recent["finish_pos"] <= 3).mean()

        # ROI for last 10 (sum of 1/odds when win, divided by N, minus 1)
        recent10 = hist.head(10)
        wins = recent10[recent10["finish_pos"] == 1]
        if len(recent10) > 0:
            odds_payoff = wins["odds_win"].dropna().sum()
            features["jockey_roi_10"] = (odds_payoff / len(recent10)) - 1.0
        else:
            features["jockey_roi_10"] = np.nan

        features["jockey_recent_wins"] = (hist.head(10)["finish_pos"] == 1).sum()

        return features

    # ------------------------------------------------------------------
    # Feature Computation — Trainer Rolling History
    # ------------------------------------------------------------------

    def _trainer_features(self, trainer_id: Optional[int], race_date: str, history_df: pd.DataFrame) -> dict:
        """Compute rolling trainer features from their recent runners."""
        null_feats = {
            "trainer_win_pct_10": np.nan,
            "trainer_place_pct_10": np.nan,
            "trainer_roi_10": np.nan,
        }

        if trainer_id is None or "trainer_id" not in history_df.columns:
            return null_feats

        hist = history_df[
            (history_df["trainer_id"] == trainer_id)
            & (history_df["date"] < race_date)
        ].sort_values("date", ascending=False)

        if hist.empty:
            return null_feats

        recent = hist.head(10)
        features = {}
        features["trainer_win_pct_10"] = (recent["finish_pos"] == 1).mean()
        features["trainer_place_pct_10"] = (recent["finish_pos"] <= 3).mean()

        # ROI
        wins = recent[recent["finish_pos"] == 1]
        odds_payoff = wins["odds_win"].dropna().sum()
        features["trainer_roi_10"] = (odds_payoff / len(recent)) - 1.0

        return features

    # ------------------------------------------------------------------
    # Feature Computation — Pace Simulation
    # ------------------------------------------------------------------

    def _get_pace_features(self, race_id: int, horse_id: int, race_df: pd.DataFrame, history_df: pd.DataFrame) -> dict:
        """
        Get pace simulation features for a horse-in-race.
        Runs PaceSimulator once per race (cached) and returns per-horse features.
        """
        from models.pace_sim import PaceSimulator, STYLE_FRONT, STYLE_STALK, STYLE_CLOSER, STYLE_DEEP

        null_feats = {
            "pace_win_prob": np.nan,
            "pace_place_prob": np.nan,
            "pace_style_front": 0,
            "pace_style_stalk": 0,
            "pace_style_closer": 0,
            "pace_style_deep": 0,
        }

        # Check cache
        if race_id not in self._pace_cache:
            # Build entries list for this race
            race_entries = race_df[race_df["race_id"] == race_id]
            if race_entries.empty:
                self._pace_cache[race_id] = {}
                return null_feats

            sim_entries = []
            distance = 2000
            for _, row in race_entries.iterrows():
                hid = row["horse_id"]
                distance = row.get("distance") or 2000

                # Get historical stats for pace simulation
                horse_hist = history_df[
                    (history_df["horse_id"] == hid)
                    & (history_df["date"] < str(row["date"]))
                ].sort_values("date", ascending=False)

                if not horse_hist.empty:
                    finishes = horse_hist["finish_pos"].dropna()
                    last_3fs = horse_hist["last_3f_secs"].dropna()
                    corners = []
                    for cp in horse_hist["corner_positions"].dropna():
                        try:
                            corners.append(int(str(cp).split("-")[0].strip()))
                        except (ValueError, IndexError):
                            pass

                    career_win_pct = (finishes == 1).mean() if len(finishes) > 0 else 0
                    last3_avg_finish = finishes.head(3).mean() if len(finishes) > 0 else None
                    avg_last_3f = last_3fs.mean() if len(last_3fs) > 0 else None
                    avg_first_corner = np.mean(corners) if corners else None
                else:
                    career_win_pct = 0
                    last3_avg_finish = None
                    avg_last_3f = None
                    avg_first_corner = None

                sim_entries.append({
                    "horse_id": hid,
                    "avg_first_corner": avg_first_corner,
                    "avg_last_3f": avg_last_3f,
                    "career_win_pct": career_win_pct,
                    "last3_avg_finish": last3_avg_finish,
                    "odds_win": row.get("odds_win"),
                })

            # Run simulation (use fewer iterations during feature building for speed)
            try:
                sim = PaceSimulator(n_simulations=2000, seed=race_id % 10000)
                pace_results = sim.simulate_race(sim_entries, distance=int(distance))
                self._pace_cache[race_id] = pace_results
            except Exception as e:
                log.debug(f"Pace sim failed for race {race_id}: {e}")
                self._pace_cache[race_id] = {}

        # Look up this horse's pace result
        pace_data = self._pace_cache.get(race_id, {}).get(horse_id)
        if pace_data is None:
            return null_feats

        style = pace_data.get("style", "")
        return {
            "pace_win_prob": pace_data.get("win_prob", np.nan),
            "pace_place_prob": pace_data.get("place_prob", np.nan),
            "pace_style_front": 1 if style == STYLE_FRONT else 0,
            "pace_style_stalk": 1 if style == STYLE_STALK else 0,
            "pace_style_closer": 1 if style == STYLE_CLOSER else 0,
            "pace_style_deep": 1 if style == STYLE_DEEP else 0,
        }

    # ------------------------------------------------------------------
    # Feature Computation — Race-Level (Static)
    # ------------------------------------------------------------------

    def _static_features(self, row: pd.Series) -> dict:
        """Compute static features that don't require historical lookups."""
        features = {}

        # Draw / post position
        features["draw"] = row.get("draw", np.nan)
        features["post_position"] = row.get("post_position", np.nan)

        # Weight
        features["weight_carried"] = row.get("weight_carried", np.nan)
        features["horse_weight"] = row.get("horse_weight", np.nan)
        features["horse_weight_change"] = row.get("horse_weight_change", 0) or 0

        # Odds (market signal)
        features["odds_win"] = row.get("odds_win", np.nan)
        features["log_odds"] = np.log(row["odds_win"]) if row.get("odds_win") and row["odds_win"] > 0 else np.nan
        features["popularity"] = row.get("popularity", np.nan)

        # Race conditions
        features["distance"] = row.get("distance", np.nan)
        features["surface_code"] = SURFACE_MAP.get(row.get("surface"), np.nan)
        features["going_code"] = GOING_MAP.get(row.get("going"), np.nan)
        features["field_size"] = row.get("field_size", np.nan)
        features["class_rank"] = CLASS_RANK.get(row.get("race_class"), CLASS_RANK.get(row.get("grade"), 10))

        # Horse attributes
        features["sex_code"] = SEX_MAP.get(row.get("sex"), np.nan)

        # Age
        if row.get("birth_year") and row.get("date"):
            try:
                race_year = int(str(row["date"])[:4])
                features["age"] = race_year - int(row["birth_year"])
            except (ValueError, TypeError):
                features["age"] = np.nan
        else:
            features["age"] = np.nan

        return features

    # ------------------------------------------------------------------
    # Normalisation — Per-Race Z-Scores
    # ------------------------------------------------------------------

    @staticmethod
    def normalise_per_race(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """
        Z-score normalise features within each race.
        This lets the model see relative strengths (e.g., "fastest in THIS field").
        """
        for col in feature_cols:
            if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
                group_mean = df.groupby("race_id")[col].transform("mean")
                group_std = df.groupby("race_id")[col].transform("std")
                # Avoid division by zero
                group_std = group_std.replace(0, 1)
                df[f"{col}_z"] = (df[col] - group_mean) / group_std

        return df

    # ------------------------------------------------------------------
    # Main Builder
    # ------------------------------------------------------------------

    def build_features_for_race(self, race_id: int) -> pd.DataFrame:
        """Build feature vectors for all entries in a single race."""
        return self._build(race_id=race_id)

    def build_features_all(self) -> pd.DataFrame:
        """Build feature vectors for all races in the database."""
        return self._build(race_id=None)

    def _build(self, race_id: Optional[int] = None) -> pd.DataFrame:
        """Core feature building logic."""
        log.info("Loading race data...")
        race_df = self._load_race_data(race_id)
        if race_df.empty:
            return pd.DataFrame()

        log.info("Loading horse/jockey history for rolling features...")
        history_df = self._load_horse_history()

        log.info(f"Building features for {len(race_df)} entries...")
        feature_rows = []

        for idx, row in race_df.iterrows():
            features = {
                "race_id": row["race_id"],
                "entry_id": row["entry_id"],
                "date": str(row["date"]) if row.get("date") else None,
                "horse_name": row.get("horse_name", ""),
            }

            # Target variable (for training)
            if row.get("finish_pos") is not None:
                features["target_win"] = 1 if row["finish_pos"] == 1 else 0
                features["target_place"] = 1 if row["finish_pos"] <= 3 else 0
                features["finish_pos"] = row["finish_pos"]

            # Static features
            features.update(self._static_features(row))

            # Rolling horse features
            features.update(
                self._horse_rolling_features(
                    horse_id=row["horse_id"],
                    race_date=str(row["date"]),
                    distance=row["distance"] or 0,
                    surface=row["surface"] or "",
                    course_id=row.get("course_id"),
                    history_df=history_df,
                )
            )

            # Jockey features
            features.update(
                self._jockey_features(
                    jockey_id=row.get("jockey_id"),
                    race_date=str(row["date"]),
                    history_df=history_df,
                )
            )

            # Trainer features
            features.update(
                self._trainer_features(
                    trainer_id=row.get("trainer_id"),
                    race_date=str(row["date"]),
                    history_df=history_df,
                )
            )

            # Pace simulation features (cached per race)
            pace_feats = self._get_pace_features(
                race_id=row["race_id"],
                horse_id=row["horse_id"],
                race_df=race_df,
                history_df=history_df,
            )
            features.update(pace_feats)

            feature_rows.append(features)

        df = pd.DataFrame(feature_rows)

        # Per-race z-score normalisation
        numeric_cols = [
            c for c in df.columns
            if c not in ["race_id", "entry_id", "target_win", "target_place", "finish_pos", "date", "horse_name"]
            and df[c].dtype in [np.float64, np.float32, np.int64, float, int]
        ]
        df = self.normalise_per_race(df, numeric_cols)

        log.info(f"✅ Built {len(df)} feature vectors with {len(df.columns)} columns")
        return df

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list[str]:
        """Return the list of feature columns (excluding IDs and targets)."""
        exclude = {"race_id", "entry_id", "target_win", "target_place", "finish_pos", "date", "horse_name"}
        return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fb = FeatureBuilder()
    df = fb.build_features_all()
    if not df.empty:
        print(f"\nFeature matrix: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSample (first 3 rows):\n{df.head(3).T}")
    else:
        print("No data — run the scraper first to populate races.")
