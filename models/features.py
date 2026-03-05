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
import re
import time
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
WEATHER_MAP = {"晴": 0, "曇": 1, "小雨": 2, "雨": 3, "小雪": 3, "雪": 3}

# ---------------------------------------------------------------------------
# Margin / beaten-lengths parser
# ---------------------------------------------------------------------------

_MARGIN_TEXT = {
    # Japanese
    "ハナ": 0.05, "クビ": 0.25, "アタマ": 0.1,
    # English (JRA EN site)
    "NS": 0.05, "NK": 0.25, "HD": 0.1,
    # Large margin / distance
    "大": 10.0, "DS": 10.0,
    # Dead heat (same finish)
    "同着": 0.0, "DH": 0.0,
}


def _parse_margin_to_lengths(val) -> float:
    """Convert a JRA margin string to numeric beaten-lengths.

    Handles: '0', '1', '1/2', '3/4', '1.1/4', '1 1/4', 'クビ', 'NK', '大', etc.
    Returns NaN for unparseable values.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan

    # Text-based margins
    if s in _MARGIN_TEXT:
        return _MARGIN_TEXT[s]
    # Compound like '1.1/4+クビ' — take the main part before '+'
    if '+' in s:
        s = s.split('+')[0].strip()

    # Pure integer or float
    try:
        return float(s)
    except ValueError:
        pass

    # Fraction only: '1/2', '3/4'
    frac_match = re.match(r'^(\d+)/(\d+)$', s)
    if frac_match:
        return int(frac_match.group(1)) / int(frac_match.group(2))

    # Mixed: '1.1/4' or '1 1/4' or '2.1/2'
    mixed_match = re.match(r'^(\d+)[.\s](\d+)/(\d+)$', s)
    if mixed_match:
        whole = int(mixed_match.group(1))
        numer = int(mixed_match.group(2))
        denom = int(mixed_match.group(3))
        return whole + numer / denom

    return np.nan

# All odds-derived features (raw + z-score normalised).
# Excluding these forces the model to learn fundamental signals only.
ODDS_FEATURES = [
    "odds_win", "log_odds", "popularity",
    "odds_win_z", "log_odds_z", "popularity_z",
    "is_longshot", "is_extreme_longshot",
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
                res.margin,
                res.first_3f_secs,
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
        # Use pre-indexed group if available, else fall back to filtering
        group = self._horse_groups.get(horse_id) if hasattr(self, '_horse_groups') else None
        if group is not None:
            hist = group[group["date"] < race_date]
        else:
            hist = history_df[
                (history_df["horse_id"] == horse_id)
                & (history_df["date"] < race_date)
            ]
        hist = hist.sort_values("date", ascending=False)

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

        # --- Class change features (Research Backlog #1) ---
        current_class_rank = CLASS_RANK.get(row.get("race_class") if hasattr(row, 'get') else None, 10) if 'row' in dir() else 10
        class_ranks = hist["race_class"].map(CLASS_RANK).dropna()
        if not class_ranks.empty:
            last_class_rank = class_ranks.iloc[0]
            current_rank = CLASS_RANK.get(
                hist.iloc[0]["race_class"] if pd.notna(hist.iloc[0].get("race_class")) else None, 10
            )
            # Negative = dropped in class (easier race), positive = rising
            features["class_change"] = current_rank - last_class_rank if len(class_ranks) >= 2 else 0
            if len(class_ranks) >= 2:
                diffs = class_ranks.diff(-1).dropna()  # sorted desc, so diff(-1) = newer - older
                recent_diffs = diffs.head(5)
                features["class_change"] = float(class_ranks.iloc[0] - class_ranks.iloc[1]) if len(class_ranks) >= 2 else 0.0
                features["class_drops_last5"] = int((recent_diffs < 0).sum())  # dropped = lower rank number = stronger
                features["class_rises_last5"] = int((recent_diffs > 0).sum())  # risen = higher rank number = weaker
            else:
                features["class_change"] = 0.0
                features["class_drops_last5"] = 0
                features["class_rises_last5"] = 0

            # Class at most recent win
            wins = hist[hist["finish_pos"] == 1]
            if not wins.empty:
                win_class = CLASS_RANK.get(wins.iloc[0]["race_class"], np.nan)
                features["class_at_last_win"] = win_class
            else:
                features["class_at_last_win"] = np.nan
        else:
            features["class_change"] = np.nan
            features["class_drops_last5"] = np.nan
            features["class_rises_last5"] = np.nan
            features["class_at_last_win"] = np.nan

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

        # --- Beaten lengths / margin features (Sprint 8.4) ---
        raw_margins = hist["margin"].dropna() if "margin" in hist.columns else pd.Series(dtype=object)
        margins = raw_margins.apply(_parse_margin_to_lengths).dropna()
        if not margins.empty:
            recent_margins = margins.head(3)
            features["beaten_lengths_avg3"] = recent_margins.mean()
            features["beaten_lengths_best"] = margins.min()  # smallest margin = best

            # Class-adjusted margin: weight margin by class quality
            # Lower class_rank = stronger race, so we scale margin down for strong races
            class_weights = hist["race_class"].map(CLASS_RANK).fillna(10)
            parsed_margins = hist["margin"].apply(_parse_margin_to_lengths)
            valid_mask = parsed_margins.notna()
            if valid_mask.any():
                weighted = parsed_margins[valid_mask] * (10.0 / class_weights[valid_mask])
                features["class_adjusted_margin"] = weighted.head(3).mean()
            else:
                features["class_adjusted_margin"] = np.nan
        else:
            features["beaten_lengths_avg3"] = np.nan
            features["beaten_lengths_best"] = np.nan
            features["class_adjusted_margin"] = np.nan

        # --- Fitness curve / layoff buckets (Sprint 8.5) ---
        days = features.get("days_since_last", np.nan)
        if not np.isnan(days) if isinstance(days, (int, float)) else days is not None:
            features["is_fresh"] = 1 if 14 <= days <= 28 else 0
            features["is_rested"] = 1 if 29 <= days <= 56 else 0
            features["is_stale"] = 1 if days > 90 else 0
        else:
            features["is_fresh"] = np.nan
            features["is_rested"] = np.nan
            features["is_stale"] = np.nan

        # --- Sectional times / early speed vs closing speed (Sprint 9.3) ---
        # Build first_3f series: use first_3f_secs where available, fall back
        # to (time_secs - last_3f_secs) for rows where it's missing.
        if "first_3f_secs" in hist.columns:
            first_3f_raw = hist["first_3f_secs"].copy()
        else:
            first_3f_raw = pd.Series(np.nan, index=hist.index)

        # Fill gaps with derived value: time_secs - last_3f_secs
        if "time_secs" in hist.columns and "last_3f_secs" in hist.columns:
            derived = hist["time_secs"] - hist["last_3f_secs"]
            # Only use derived where both components are valid & positive
            valid_derived = derived[(hist["time_secs"].notna()) & (hist["last_3f_secs"].notna()) & (hist["time_secs"] > 0) & (hist["last_3f_secs"] > 0)]
            first_3f_raw = first_3f_raw.fillna(valid_derived)

        first_3f = first_3f_raw.dropna()
        if not first_3f.empty:
            features["avg_first_3f"] = first_3f.mean()
            features["best_first_3f"] = first_3f.min()
        else:
            features["avg_first_3f"] = np.nan
            features["best_first_3f"] = np.nan

        # Derive early speed = time before last 3F (works even without first_3f_secs)
        if "time_secs" in hist.columns and "last_3f_secs" in hist.columns:
            valid_mask = hist["time_secs"].notna() & hist["last_3f_secs"].notna() & (hist["time_secs"] > 0) & (hist["last_3f_secs"] > 0)
            valid = hist[valid_mask]
            if not valid.empty:
                early_times = valid["time_secs"] - valid["last_3f_secs"]
                features["avg_early_speed"] = early_times.mean()
                # Early-to-late ratio: lower = more front-loaded
                ratios = early_times / valid["last_3f_secs"]
                features["early_late_ratio_avg3"] = ratios.head(3).mean()
            else:
                features["avg_early_speed"] = np.nan
                features["early_late_ratio_avg3"] = np.nan
        else:
            features["avg_early_speed"] = np.nan
            features["early_late_ratio_avg3"] = np.nan

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
            # Research backlog: class change
            "class_change", "class_drops_last5", "class_rises_last5", "class_at_last_win",
            # Sprint 7 features
            "weather_code", "going_x_surface", "going_x_distance",
            "horse_going_win_pct", "horse_wet_track_advantage",
            # Research backlog: weather refinement
            "horse_heavy_speed_diff", "going_x_dist_x_surface",
            "draw_bias_at_course", "draw_low_win_pct", "draw_high_win_pct",
            "draw_bias_score", "course_month_bias",
            "sire_runners", "sire_win_pct", "sire_win_pct_surface",
            "sire_win_pct_distance", "sire_avg_finish",
            # Trainer-based pedigree fallback
            "trainer_offspring_win_pct", "trainer_offspring_avg_finish",
            # Research backlog: course × jockey
            "jockey_course_runs", "jockey_course_win_pct", "jockey_course_place_pct",

            # Cross-sectional odds features (always populated when odds exist)
            "odds_rank", "odds_ratio_to_fav", "odds_deviation",
            # Sprint 8: beaten lengths
            "beaten_lengths_avg3", "beaten_lengths_best", "class_adjusted_margin",
            # Sprint 8: fitness curve
            "is_fresh", "is_rested", "is_stale",
            # Sprint 9: sectional times
            "avg_first_3f", "best_first_3f", "avg_early_speed", "early_late_ratio_avg3",
            # Sprint 8: speed figures
            "speed_figure_last", "speed_figure_best", "speed_figure_avg3",
            # Sprint 8: jockey-trainer combo
            "jt_combo_runs", "jt_combo_win_pct", "jt_combo_place_pct",
            # Sprint 8: field quality
            "field_avg_career_win_pct", "horse_vs_field_quality",
            # Sprint 8: weight vs field
            "weight_vs_field_avg", "weight_per_kg_body",
            # Sprint 8: age × class
            "age_x_class", "is_improving_3yo",
            # Sprint 9: seasonal form
            "month_of_year", "horse_month_win_pct", "season_code",
            # Sprint 9: broodmare sire
            "bms_runners", "bms_win_pct", "bms_win_pct_surface", "bms_win_pct_distance",
            # Sprint 9: enhanced track bias
            "draw_bias_90d",
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

        # Use pre-indexed group if available
        group = self._jockey_groups.get(jockey_id) if hasattr(self, '_jockey_groups') else None
        if group is not None:
            hist = group[group["date"] < race_date]
        else:
            hist = history_df[
                (history_df["jockey_id"] == jockey_id)
                & (history_df["date"] < race_date)
            ]
        hist = hist.sort_values("date", ascending=False)

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
            "trainer_14d_runs": np.nan,
            "trainer_14d_win_pct": np.nan,
            "trainer_14d_place_pct": np.nan,
        }

        if trainer_id is None or "trainer_id" not in history_df.columns:
            return null_feats

        # Use pre-indexed group if available
        group = self._trainer_groups.get(trainer_id) if hasattr(self, '_trainer_groups') else None
        if group is not None:
            hist = group[group["date"] < race_date]
        else:
            hist = history_df[
                (history_df["trainer_id"] == trainer_id)
                & (history_df["date"] < race_date)
            ]
        hist = hist.sort_values("date", ascending=False)

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

        # --- Trainer last-14-day form (Research Backlog #2) ---
        try:
            cutoff = str(pd.to_datetime(race_date) - pd.Timedelta(days=14))
            recent_14d = hist[hist["date"] >= cutoff]
            features["trainer_14d_runs"] = len(recent_14d)
            if len(recent_14d) > 0:
                features["trainer_14d_win_pct"] = (recent_14d["finish_pos"] == 1).mean()
                features["trainer_14d_place_pct"] = (recent_14d["finish_pos"] <= 3).mean()
            else:
                features["trainer_14d_win_pct"] = np.nan
                features["trainer_14d_place_pct"] = np.nan
        except Exception:
            features["trainer_14d_runs"] = np.nan
            features["trainer_14d_win_pct"] = np.nan
            features["trainer_14d_place_pct"] = np.nan

        return features

    # ------------------------------------------------------------------
    # Feature Computation — Course × Jockey Interaction (Research Backlog #3)
    # ------------------------------------------------------------------

    def _course_jockey_features(
        self, jockey_id: Optional[int], course_id: Optional[int],
        race_date: str, history_df: pd.DataFrame
    ) -> dict:
        """Compute course-specific jockey performance. Specialist jockeys dominate certain courses."""
        null_feats = {
            "jockey_course_runs": np.nan,
            "jockey_course_win_pct": np.nan,
            "jockey_course_place_pct": np.nan,
        }

        if jockey_id is None or course_id is None or "course_id" not in history_df.columns:
            return null_feats

        # Use pre-indexed group if available
        group = self._jockey_groups.get(jockey_id) if hasattr(self, '_jockey_groups') else None
        if group is not None:
            hist = group[
                (group["course_id"] == course_id)
                & (group["date"] < race_date)
            ]
        else:
            hist = history_df[
                (history_df["jockey_id"] == jockey_id)
                & (history_df["course_id"] == course_id)
                & (history_df["date"] < race_date)
            ]

        if hist.empty:
            return null_feats

        features = {}
        features["jockey_course_runs"] = len(hist)
        features["jockey_course_win_pct"] = (hist["finish_pos"] == 1).mean()
        features["jockey_course_place_pct"] = (hist["finish_pos"] <= 3).mean()

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
                # Use pre-indexed group if available (pace sim)
                group = self._horse_groups.get(hid) if hasattr(self, '_horse_groups') else None
                if group is not None:
                    horse_hist = group[group["date"] < str(row["date"])].sort_values("date", ascending=False)
                else:
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



    def _cross_sectional_odds_features(
        self, race_id: int, odds_win: Optional[float], race_df: pd.DataFrame
    ) -> dict:
        """
        Compute cross-sectional odds features from the within-race odds
        distribution. Works with a single odds value per entry (no time series
        needed), so these are always populated when odds exist.

        Features:
            odds_rank: rank of this horse's odds within the race (1 = favourite)
            odds_ratio_to_fav: this horse's odds / favourite odds (1.0 = is favourite)
            odds_deviation: (log_odds - field median log_odds) / field std
        """
        null_feats = {
            "odds_rank": np.nan,
            "odds_ratio_to_fav": np.nan,
            "odds_deviation": np.nan,
        }

        if odds_win is None or odds_win <= 0:
            return null_feats

        # Build per-race odds cache on first call
        if not hasattr(self, "_race_odds_cache"):
            self._race_odds_cache = {}

        if race_id not in self._race_odds_cache:
            # Collect all odds for this race
            race_entries = race_df[race_df["race_id"] == race_id]
            race_odds = race_entries["odds_win"].dropna()
            race_odds = race_odds[race_odds > 0]
            if race_odds.empty:
                self._race_odds_cache[race_id] = None
            else:
                sorted_odds = race_odds.sort_values().values
                log_odds = np.log(race_odds.values)
                self._race_odds_cache[race_id] = {
                    "sorted": sorted_odds,
                    "fav_odds": sorted_odds[0],
                    "log_median": np.median(log_odds),
                    "log_std": np.std(log_odds) if len(log_odds) > 1 else 1.0,
                }

        cache = self._race_odds_cache.get(race_id)
        if cache is None:
            return null_feats

        # Rank: 1 = shortest odds (favourite)
        rank = int((cache["sorted"] <= odds_win).sum())
        features = {
            "odds_rank": rank,
            "odds_ratio_to_fav": odds_win / cache["fav_odds"] if cache["fav_odds"] > 0 else np.nan,
            "odds_deviation": (
                (np.log(odds_win) - cache["log_median"]) / cache["log_std"]
                if cache["log_std"] > 0 else 0.0
            ),
        }
        return features

    # ------------------------------------------------------------------
    # Feature Computation — Weather Interactions (Sprint 7.2)
    # ------------------------------------------------------------------

    def _weather_interaction_features(
        self, row: pd.Series, history_df: pd.DataFrame
    ) -> dict:
        """
        Compute weather × surface × distance interaction features.
        Some horses are 'heavy track specialists' — this captures that.
        """
        features = {}
        weather = row.get("weather")
        going = row.get("going")
        surface = row.get("surface")
        distance = row.get("distance") or 0
        horse_id = row.get("horse_id")
        race_date = str(row.get("date", ""))

        # Encode weather
        features["weather_code"] = WEATHER_MAP.get(weather, np.nan)

        # Interaction: going × surface (dirt handles rain differently)
        going_code = GOING_MAP.get(going, np.nan)
        surface_code = SURFACE_MAP.get(surface, np.nan)
        features["going_x_surface"] = (
            going_code * surface_code
            if not (np.isnan(going_code) if isinstance(going_code, float) else False)
            and not (np.isnan(surface_code) if isinstance(surface_code, float) else False)
            else np.nan
        )

        # Interaction: going × distance (heavy going hurts more in long races)
        features["going_x_distance"] = (
            going_code * (distance / 1000.0)
            if not (np.isnan(going_code) if isinstance(going_code, float) else False)
            and distance > 0
            else np.nan
        )

        # Horse-specific going performance from history
        # Use pre-indexed group if available
        if horse_id is not None:
            group = self._horse_groups.get(horse_id) if hasattr(self, '_horse_groups') else None
            if group is not None:
                hist = group[group["date"] < race_date]
            else:
                hist = history_df[
                    (history_df["horse_id"] == horse_id)
                    & (history_df["date"] < race_date)
                ]
        else:
            hist = pd.DataFrame()

        if not hist.empty and "going" in hist.columns:
            # Win% on current going type
            going_hist = hist[hist["going"] == going]
            features["horse_going_win_pct"] = (
                (going_hist["finish_pos"] == 1).mean()
                if len(going_hist) > 0 else np.nan
            )

            # Wet track advantage: win% on heavy/bad − win% on good
            good = hist[hist["going"] == "良"]
            wet = hist[hist["going"].isin(["重", "不良", "稍重"])]
            good_wr = (good["finish_pos"] == 1).mean() if len(good) >= 2 else np.nan
            wet_wr = (wet["finish_pos"] == 1).mean() if len(wet) >= 2 else np.nan
            features["horse_wet_track_advantage"] = (
                wet_wr - good_wr
                if not np.isnan(wet_wr) and not np.isnan(good_wr)
                else np.nan
            )

            # --- Weather refinement (Research Backlog #5) ---
            # Speed differential on heavy vs good ground
            good_times = good["time_secs"].dropna()
            wet_times = wet["time_secs"].dropna()
            if len(good_times) >= 2 and len(wet_times) >= 2:
                features["horse_heavy_speed_diff"] = wet_times.mean() - good_times.mean()
            else:
                features["horse_heavy_speed_diff"] = np.nan
        else:
            features["horse_going_win_pct"] = np.nan
            features["horse_wet_track_advantage"] = np.nan
            features["horse_heavy_speed_diff"] = np.nan

        # Three-way interaction: going × distance × surface
        features["going_x_dist_x_surface"] = (
            going_code * (distance / 1000.0) * surface_code
            if not (np.isnan(going_code) if isinstance(going_code, float) else False)
            and not (np.isnan(surface_code) if isinstance(surface_code, float) else False)
            and distance > 0
            else np.nan
        )

        return features

    # ------------------------------------------------------------------
    # Feature Computation — Track Bias (Sprint 7.3)
    # ------------------------------------------------------------------

    def _track_bias_features(
        self, row: pd.Series, history_df: pd.DataFrame
    ) -> dict:
        """
        Compute draw/track bias features from historical results.
        Inside/outside draw advantage varies by course and month.
        """
        features = {}
        course_id = row.get("course_id")
        draw = row.get("draw")
        race_date = str(row.get("date", ""))
        race_month = None
        try:
            race_month = int(race_date[5:7]) if len(race_date) >= 7 else None
        except (ValueError, TypeError):
            pass

        if course_id is None or "course_id" not in history_df.columns:
            features["draw_bias_at_course"] = np.nan
            features["draw_low_win_pct"] = np.nan
            features["draw_high_win_pct"] = np.nan
            features["draw_bias_score"] = np.nan
            features["course_month_bias"] = np.nan
            return features

        # Historical data at this course BEFORE this race
        # Use pre-indexed group if available
        group = self._course_groups.get(course_id) if hasattr(self, '_course_groups') else None
        if group is not None:
            course_hist = group[group["date"] < race_date]
        else:
            course_hist = history_df[
                (history_df["course_id"] == course_id)
                & (history_df["date"] < race_date)
            ]

        if course_hist.empty or "draw" not in course_hist.columns:
            features["draw_bias_at_course"] = np.nan
            features["draw_low_win_pct"] = np.nan
            features["draw_high_win_pct"] = np.nan
            features["draw_bias_score"] = np.nan
            features["course_month_bias"] = np.nan
            return features

        # Avg finish position for this draw at this course
        if draw is not None:
            draw_hist = course_hist[course_hist["draw"] == draw]
            features["draw_bias_at_course"] = (
                draw_hist["finish_pos"].mean() if len(draw_hist) >= 3 else np.nan
            )
        else:
            features["draw_bias_at_course"] = np.nan

        # Inner draws (1-4) vs outer draws (9+) win rates
        inner = course_hist[course_hist["draw"].between(1, 4)]
        outer = course_hist[course_hist["draw"] >= 9]
        inner_wr = (inner["finish_pos"] == 1).mean() if len(inner) >= 10 else np.nan
        outer_wr = (outer["finish_pos"] == 1).mean() if len(outer) >= 10 else np.nan
        features["draw_low_win_pct"] = inner_wr
        features["draw_high_win_pct"] = outer_wr

        # Bias score: positive means horse's draw is advantaged
        if draw is not None and not np.isnan(inner_wr) and not np.isnan(outer_wr):
            if draw <= 4:
                features["draw_bias_score"] = inner_wr - outer_wr
            elif draw >= 9:
                features["draw_bias_score"] = outer_wr - inner_wr
            else:
                features["draw_bias_score"] = 0.0  # middle draws
        else:
            features["draw_bias_score"] = np.nan

        # Seasonal bias: avg finish for this draw in this calendar month
        if draw is not None and race_month is not None:
            month_draw_hist = course_hist[
                (course_hist["draw"] == draw)
            ]
            # Extract month from date strings
            if not month_draw_hist.empty:
                try:
                    months = month_draw_hist["date"].str[5:7].astype(int)
                    same_month = month_draw_hist[months == race_month]
                    features["course_month_bias"] = (
                        same_month["finish_pos"].mean()
                        if len(same_month) >= 3 else np.nan
                    )
                except Exception:
                    features["course_month_bias"] = np.nan
            else:
                features["course_month_bias"] = np.nan
        else:
            features["course_month_bias"] = np.nan

        # --- Rolling 90-day draw bias (Sprint 9.4) ---
        if draw is not None and race_date and not course_hist.empty:
            try:
                cutoff = str(pd.to_datetime(race_date) - pd.Timedelta(days=90))[:10]
                recent_90d = course_hist[course_hist["date"] >= cutoff]
                if not recent_90d.empty and "draw" in recent_90d.columns:
                    draw_90d = recent_90d[recent_90d["draw"] == draw]
                    features["draw_bias_90d"] = (
                        draw_90d["finish_pos"].mean()
                        if len(draw_90d) >= 3 else np.nan
                    )
                else:
                    features["draw_bias_90d"] = np.nan
            except Exception:
                features["draw_bias_90d"] = np.nan
        else:
            features["draw_bias_90d"] = np.nan

        return features

    # ------------------------------------------------------------------
    # Feature Computation — Pedigree (Sprint 7.4)
    # ------------------------------------------------------------------

    def _pedigree_features(
        self, horse_id: int, race_date: str, distance: int,
        surface: str, trainer_id: Optional[int],
        history_df: pd.DataFrame
    ) -> dict:
        """
        Compute sire-based pedigree features.
        Uses sire_name matching against historical results to find
        sibling performance patterns.

        Falls back to trainer-based proxy features when sire is unknown:
        trainers specialise in certain bloodlines, so the trainer's stable
        win rate acts as a weak pedigree signal.
        """
        null_sire = {
            "sire_runners": np.nan,
            "sire_win_pct": np.nan,
            "sire_win_pct_surface": np.nan,
            "sire_win_pct_distance": np.nan,
            "sire_avg_finish": np.nan,
        }
        null_trainer_proxy = {
            "trainer_offspring_win_pct": np.nan,
            "trainer_offspring_avg_finish": np.nan,
        }

        features = {}

        # Look up sire name for this horse from DB
        sire_name = self._get_sire_name(horse_id)
        has_sire = False

        if sire_name:
            # Find all offspring of the same sire in history
            sibling_ids = self._get_sire_offspring(sire_name, horse_id)
            if sibling_ids:
                # Filter history to siblings only, before current race
                sib_hist = history_df[
                    (history_df["horse_id"].isin(sibling_ids))
                    & (history_df["date"] < race_date)
                ]

                if not sib_hist.empty:
                    has_sire = True
                    features["sire_runners"] = len(sib_hist)
                    features["sire_win_pct"] = (sib_hist["finish_pos"] == 1).mean()
                    features["sire_avg_finish"] = sib_hist["finish_pos"].mean()

                    # Sire × surface affinity
                    surf_hist = sib_hist[sib_hist["surface"] == surface]
                    features["sire_win_pct_surface"] = (
                        (surf_hist["finish_pos"] == 1).mean()
                        if len(surf_hist) >= 3 else np.nan
                    )

                    # Sire × distance affinity (±200m)
                    dist_hist = sib_hist[
                        (sib_hist["distance"] >= distance - 200)
                        & (sib_hist["distance"] <= distance + 200)
                    ]
                    features["sire_win_pct_distance"] = (
                        (dist_hist["finish_pos"] == 1).mean()
                        if len(dist_hist) >= 3 else np.nan
                    )

        if not has_sire:
            features.update(null_sire)

        # --- Trainer-based pedigree fallback ---
        # When sire is unknown, use trainer's overall stable performance.
        # Trainers specialise in certain bloodlines and produce correlated results.
        if trainer_id is not None and "trainer_id" in history_df.columns:
            group = self._trainer_groups.get(trainer_id) if hasattr(self, '_trainer_groups') else None
            if group is not None:
                trainer_hist = group[group["date"] < race_date]
            else:
                trainer_hist = history_df[
                    (history_df["trainer_id"] == trainer_id)
                    & (history_df["date"] < race_date)
                ]
            if not trainer_hist.empty and len(trainer_hist) >= 5:
                features["trainer_offspring_win_pct"] = (trainer_hist["finish_pos"] == 1).mean()
                features["trainer_offspring_avg_finish"] = trainer_hist["finish_pos"].mean()
            else:
                features.update(null_trainer_proxy)
        else:
            features.update(null_trainer_proxy)

        return features

    def _get_sire_name(self, horse_id: int) -> Optional[str]:
        """Look up sire_name from DB cache."""
        if not hasattr(self, "_sire_cache"):
            self._sire_cache = {}
            try:
                with get_session() as session:
                    rows = session.execute(
                        text("SELECT id, sire_name FROM horses WHERE sire_name IS NOT NULL")
                    ).fetchall()
                    self._sire_cache = {r[0]: r[1] for r in rows}
            except Exception:
                pass
        return self._sire_cache.get(horse_id)

    def _get_sire_offspring(self, sire_name: str, exclude_horse_id: int) -> list[int]:
        """Find all horse IDs that share the same sire_name."""
        if not hasattr(self, "_offspring_cache"):
            self._offspring_cache = {}
            try:
                with get_session() as session:
                    rows = session.execute(
                        text("SELECT id, sire_name FROM horses WHERE sire_name IS NOT NULL")
                    ).fetchall()
                    for hid, sname in rows:
                        self._offspring_cache.setdefault(sname, []).append(hid)
            except Exception:
                pass

        offspring = self._offspring_cache.get(sire_name, [])
        return [h for h in offspring if h != exclude_horse_id]

    # ------------------------------------------------------------------
    # Feature Computation — Seasonal Form (Sprint 9.1)
    # ------------------------------------------------------------------

    def _seasonal_form_features(
        self, horse_id: int, race_date: str, history_df: pd.DataFrame
    ) -> dict:
        """
        Compute seasonal/monthly form patterns.
        Some horses peak in specific seasons or months.
        """
        features = {}

        # Extract month from race date
        try:
            month = int(race_date[5:7])
            features["month_of_year"] = month
            # Season: Spring(3-5)=0, Summer(6-8)=1, Autumn(9-11)=2, Winter(12-2)=3
            if month in (3, 4, 5):
                features["season_code"] = 0
            elif month in (6, 7, 8):
                features["season_code"] = 1
            elif month in (9, 10, 11):
                features["season_code"] = 2
            else:
                features["season_code"] = 3
        except (ValueError, TypeError, IndexError):
            features["month_of_year"] = np.nan
            features["season_code"] = np.nan

        # Horse-specific monthly win rate
        group = self._horse_groups.get(horse_id) if hasattr(self, '_horse_groups') else None
        if group is not None and not group.empty:
            hist = group[group["date"] < race_date]
            if not hist.empty and "date" in hist.columns:
                try:
                    month_val = features.get("month_of_year")
                    if month_val is not None and not np.isnan(month_val):
                        months = hist["date"].str[5:7].astype(int)
                        same_month = hist[months == int(month_val)]
                        if len(same_month) >= 2:
                            features["horse_month_win_pct"] = (
                                (same_month["finish_pos"] == 1).mean()
                            )
                        else:
                            features["horse_month_win_pct"] = np.nan
                    else:
                        features["horse_month_win_pct"] = np.nan
                except Exception:
                    features["horse_month_win_pct"] = np.nan
            else:
                features["horse_month_win_pct"] = np.nan
        else:
            features["horse_month_win_pct"] = np.nan

        return features

    # ------------------------------------------------------------------
    # Feature Computation — Broodmare Sire (Sprint 9.2)
    # ------------------------------------------------------------------

    def _broodmare_sire_features(
        self, horse_id: int, race_date: str, distance: int,
        surface: str, history_df: pd.DataFrame
    ) -> dict:
        """
        Compute broodmare sire (母父) affinity features.
        The dam's sire is equally important in Japanese racing
        for surface and distance aptitude.
        """
        null_feats = {
            "bms_runners": np.nan,
            "bms_win_pct": np.nan,
            "bms_win_pct_surface": np.nan,
            "bms_win_pct_distance": np.nan,
        }

        bms_name = self._get_broodmare_sire_name(horse_id)
        if not bms_name:
            return null_feats

        # Find all offspring with the same broodmare sire
        bms_offspring = self._get_bms_offspring(bms_name, horse_id)
        if not bms_offspring:
            return null_feats

        sib_hist = history_df[
            (history_df["horse_id"].isin(bms_offspring))
            & (history_df["date"] < race_date)
        ]

        if sib_hist.empty:
            return null_feats

        features = {
            "bms_runners": len(sib_hist),
            "bms_win_pct": (sib_hist["finish_pos"] == 1).mean(),
        }

        # BMS × surface affinity
        surf_hist = sib_hist[sib_hist["surface"] == surface]
        features["bms_win_pct_surface"] = (
            (surf_hist["finish_pos"] == 1).mean()
            if len(surf_hist) >= 3 else np.nan
        )

        # BMS × distance affinity (±200m)
        dist_hist = sib_hist[
            (sib_hist["distance"] >= distance - 200)
            & (sib_hist["distance"] <= distance + 200)
        ]
        features["bms_win_pct_distance"] = (
            (dist_hist["finish_pos"] == 1).mean()
            if len(dist_hist) >= 3 else np.nan
        )

        return features

    def _get_broodmare_sire_name(self, horse_id: int) -> Optional[str]:
        """Look up broodmare sire name via broodmare_sire_id → sire_name."""
        if not hasattr(self, "_bms_name_cache"):
            self._bms_name_cache = {}
            try:
                with get_session() as session:
                    # Join horse to its broodmare sire's name
                    rows = session.execute(text(
                        "SELECT h.id, bms.sire_name "
                        "FROM horses h "
                        "JOIN horses bms ON bms.id = h.broodmare_sire_id "
                        "WHERE h.broodmare_sire_id IS NOT NULL "
                        "AND bms.sire_name IS NOT NULL"
                    )).fetchall()
                    self._bms_name_cache = {r[0]: r[1] for r in rows}
            except Exception:
                pass
        return self._bms_name_cache.get(horse_id)

    def _get_bms_offspring(self, bms_name: str, exclude_horse_id: int) -> list[int]:
        """Find all horse IDs that share the same broodmare sire name."""
        if not hasattr(self, "_bms_offspring_cache"):
            self._bms_offspring_cache = {}
            try:
                with get_session() as session:
                    rows = session.execute(text(
                        "SELECT h.id, bms.sire_name "
                        "FROM horses h "
                        "JOIN horses bms ON bms.id = h.broodmare_sire_id "
                        "WHERE h.broodmare_sire_id IS NOT NULL "
                        "AND bms.sire_name IS NOT NULL"
                    )).fetchall()
                    for hid, sname in rows:
                        self._bms_offspring_cache.setdefault(sname, []).append(hid)
            except Exception:
                pass

        offspring = self._bms_offspring_cache.get(bms_name, [])
        return [h for h in offspring if h != exclude_horse_id]

    # ------------------------------------------------------------------
    # Feature Computation — Speed Figures (Sprint 8.1)
    # ------------------------------------------------------------------

    def _load_speed_baselines(self, history_df: pd.DataFrame):
        """Pre-compute median times per (course_id, distance, going) for speed figures."""
        if hasattr(self, "_speed_baselines"):
            return

        valid = history_df[
            history_df["time_secs"].notna()
            & history_df["time_secs"].gt(0)
            & history_df["course_id"].notna()
            & history_df["distance"].notna()
            & history_df["going"].notna()
        ].copy()

        if valid.empty:
            self._speed_baselines = {}
            return

        grouped = valid.groupby(["course_id", "distance", "going"])["time_secs"]
        self._speed_baselines = grouped.median().to_dict()
        log.info(f"Speed baselines: {len(self._speed_baselines)} (course, dist, going) combos")

    def _speed_figure_features(
        self, horse_id: int, race_date: str, distance: int,
        course_id, going: str, history_df: pd.DataFrame
    ) -> dict:
        """
        Compute normalised speed figures from historical race times.
        Speed figure = (baseline_time - horse_time) / baseline_time * 1000
        Positive = faster than baseline. Higher = better.
        """
        null_feats = {
            "speed_figure_last": np.nan,
            "speed_figure_best": np.nan,
            "speed_figure_avg3": np.nan,
        }

        self._load_speed_baselines(history_df)
        if not self._speed_baselines:
            return null_feats

        # Get horse history
        group = self._horse_groups.get(horse_id) if hasattr(self, '_horse_groups') else None
        if group is not None:
            hist = group[group["date"] < race_date]
        else:
            hist = history_df[
                (history_df["horse_id"] == horse_id)
                & (history_df["date"] < race_date)
            ]
        hist = hist.sort_values("date", ascending=False)

        if hist.empty:
            return null_feats

        # Compute speed figure for each past race
        figures = []
        for _, row in hist.iterrows():
            t = row.get("time_secs")
            d = row.get("distance")
            c = row.get("course_id")
            g = row.get("going")
            if t and t > 0 and d and c and g:
                baseline = self._speed_baselines.get((c, d, g))
                if baseline and baseline > 0:
                    fig = (baseline - t) / baseline * 1000
                    figures.append(fig)

        if not figures:
            return null_feats

        return {
            "speed_figure_last": figures[0],
            "speed_figure_best": max(figures),
            "speed_figure_avg3": np.mean(figures[:3]),
        }

    # ------------------------------------------------------------------
    # Feature Computation — Jockey-Trainer Combo (Sprint 8.2)
    # ------------------------------------------------------------------

    def _jockey_trainer_combo_features(
        self, jockey_id, trainer_id, race_date: str,
        history_df: pd.DataFrame
    ) -> dict:
        """
        Compute historical win/place rate for a specific jockey-trainer pair.
        Some jockeys perform significantly better for certain trainers.
        """
        null_feats = {
            "jt_combo_runs": np.nan,
            "jt_combo_win_pct": np.nan,
            "jt_combo_place_pct": np.nan,
        }

        if jockey_id is None or trainer_id is None:
            return null_feats

        # Build combo index on first call
        if not hasattr(self, "_jt_combo_groups"):
            if "jockey_id" in history_df.columns and "trainer_id" in history_df.columns:
                valid = history_df.dropna(subset=["jockey_id", "trainer_id"])
                self._jt_combo_groups = dict(list(
                    valid.groupby(["jockey_id", "trainer_id"])
                ))
            else:
                self._jt_combo_groups = {}

        group = self._jt_combo_groups.get((jockey_id, trainer_id))
        if group is None or group.empty:
            return null_feats

        hist = group[group["date"] < race_date]
        if hist.empty:
            return null_feats

        return {
            "jt_combo_runs": len(hist),
            "jt_combo_win_pct": (hist["finish_pos"] == 1).mean(),
            "jt_combo_place_pct": (hist["finish_pos"] <= 3).mean(),
        }

    # ------------------------------------------------------------------
    # Feature Computation — Field Quality (Sprint 8.3)
    # ------------------------------------------------------------------

    def _field_quality_features(
        self, race_id: int, horse_career_win_pct: float,
        race_df: pd.DataFrame, history_df: pd.DataFrame
    ) -> dict:
        """
        Rate the overall strength of this race's field.
        A horse's finish in a strong field is worth more than in a weak one.
        """
        null_feats = {
            "field_avg_career_win_pct": np.nan,
            "horse_vs_field_quality": np.nan,
        }

        # Cache field quality per race
        if not hasattr(self, "_field_quality_cache"):
            self._field_quality_cache = {}

        if race_id not in self._field_quality_cache:
            race_entries = race_df[race_df["race_id"] == race_id]
            horse_ids = race_entries["horse_id"].unique()

            # Compute career win pct for each horse from history
            win_pcts = []
            for hid in horse_ids:
                group = self._horse_groups.get(hid) if hasattr(self, '_horse_groups') else None
                if group is not None and not group.empty:
                    finishes = group["finish_pos"].dropna()
                    if len(finishes) > 0:
                        win_pcts.append((finishes == 1).mean())

            if win_pcts:
                self._field_quality_cache[race_id] = np.mean(win_pcts)
            else:
                self._field_quality_cache[race_id] = None

        field_avg = self._field_quality_cache.get(race_id)
        if field_avg is None:
            return null_feats

        return {
            "field_avg_career_win_pct": field_avg,
            "horse_vs_field_quality": (
                horse_career_win_pct - field_avg
                if not np.isnan(horse_career_win_pct) else np.nan
            ),
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
        odds = row.get("odds_win", np.nan)
        features["odds_win"] = odds
        features["log_odds"] = np.log(odds) if odds and odds > 0 else np.nan
        features["popularity"] = row.get("popularity", np.nan)

        # Longshot flags (binary) — helps model learn to discount extreme odds
        features["is_longshot"] = 1 if odds and odds > 30 else 0
        features["is_extreme_longshot"] = 1 if odds and odds > 50 else 0

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
        t0 = time.time()
        log.info("Loading race data...")
        race_df = self._load_race_data(race_id)
        if race_df.empty:
            return pd.DataFrame()

        log.info("Loading horse/jockey history for rolling features...")
        history_df = self._load_horse_history()

        # Pre-index history by entity for O(1) lookups (major speedup)
        log.info("Pre-indexing history for fast lookups...")
        self._horse_groups = dict(list(history_df.groupby("horse_id")))
        self._jockey_groups = (
            dict(list(history_df.groupby("jockey_id")))
            if "jockey_id" in history_df.columns else {}
        )
        self._trainer_groups = (
            dict(list(history_df.groupby("trainer_id")))
            if "trainer_id" in history_df.columns else {}
        )
        self._course_groups = (
            dict(list(history_df.groupby("course_id")))
            if "course_id" in history_df.columns else {}
        )
        log.info(
            f"Indexed: {len(self._horse_groups)} horses, "
            f"{len(self._jockey_groups)} jockeys, "
            f"{len(self._trainer_groups)} trainers, "
            f"{len(self._course_groups)} courses"
        )

        total = len(race_df)
        log.info(f"Building features for {total} entries...")
        feature_rows = []
        t_loop = time.time()

        for i, (idx, row) in enumerate(race_df.iterrows()):
            if i > 0 and i % 1000 == 0:
                elapsed = time.time() - t_loop
                rate = i / elapsed
                eta = (total - i) / rate
                log.info(
                    f"  [{i:,}/{total:,}] {i/total*100:.0f}% | "
                    f"{rate:.0f} entries/s | ETA {eta:.0f}s"
                )
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

            # Weather interaction features (Sprint 7.2)
            features.update(
                self._weather_interaction_features(
                    row=row,
                    history_df=history_df,
                )
            )

            # Track bias features (Sprint 7.3)
            features.update(
                self._track_bias_features(
                    row=row,
                    history_df=history_df,
                )
            )

            # Pedigree features (Sprint 7.4) — with trainer fallback
            features.update(
                self._pedigree_features(
                    horse_id=row["horse_id"],
                    race_date=str(row["date"]),
                    distance=row["distance"] or 0,
                    surface=row["surface"] or "",
                    trainer_id=row.get("trainer_id"),
                    history_df=history_df,
                )
            )

            # Course × jockey interaction (Research Backlog #3)
            features.update(
                self._course_jockey_features(
                    jockey_id=row.get("jockey_id"),
                    course_id=row.get("course_id"),
                    race_date=str(row["date"]),
                    history_df=history_df,
                )
            )



            # Cross-sectional odds features (always populated)
            features.update(
                self._cross_sectional_odds_features(
                    race_id=row["race_id"],
                    odds_win=row.get("odds_win"),
                    race_df=race_df,
                )
            )

            # --- Sprint 8: New domain-specific features ---

            # Speed figures (normalised time by course/distance/going)
            features.update(
                self._speed_figure_features(
                    horse_id=row["horse_id"],
                    race_date=str(row["date"]),
                    distance=row["distance"] or 0,
                    course_id=row.get("course_id"),
                    going=row.get("going") or "",
                    history_df=history_df,
                )
            )

            # Jockey-trainer combo win rates
            features.update(
                self._jockey_trainer_combo_features(
                    jockey_id=row.get("jockey_id"),
                    trainer_id=row.get("trainer_id"),
                    race_date=str(row["date"]),
                    history_df=history_df,
                )
            )

            # Field quality index
            features.update(
                self._field_quality_features(
                    race_id=row["race_id"],
                    horse_career_win_pct=features.get("career_win_pct", np.nan),
                    race_df=race_df,
                    history_df=history_df,
                )
            )

            # Weight carried vs. field average
            if not hasattr(self, "_weight_field_cache"):
                self._weight_field_cache = {}
            rid = row["race_id"]
            if rid not in self._weight_field_cache:
                wc = race_df[race_df["race_id"] == rid]["weight_carried"].dropna()
                self._weight_field_cache[rid] = wc.mean() if len(wc) > 0 else None
            field_avg_wt = self._weight_field_cache.get(rid)
            wt = row.get("weight_carried")
            hw = row.get("horse_weight")
            features["weight_vs_field_avg"] = (
                wt - field_avg_wt if wt is not None and field_avg_wt is not None else np.nan
            )
            features["weight_per_kg_body"] = (
                wt / hw if wt and hw and hw > 0 else np.nan
            )

            # Age × class interaction
            age = features.get("age", np.nan)
            class_rank = features.get("class_rank", np.nan)
            class_change = features.get("class_change", np.nan)
            if not np.isnan(age) and not np.isnan(class_rank):
                features["age_x_class"] = age * class_rank
            else:
                features["age_x_class"] = np.nan
            if not np.isnan(age) and not np.isnan(class_change):
                features["is_improving_3yo"] = 1 if age == 3 and class_change < 0 else 0
            else:
                features["is_improving_3yo"] = np.nan

            # --- Sprint 9: New features ---

            # Seasonal / monthly form patterns
            features.update(
                self._seasonal_form_features(
                    horse_id=row["horse_id"],
                    race_date=str(row["date"]),
                    history_df=history_df,
                )
            )

            # Broodmare sire (母父) affinity
            features.update(
                self._broodmare_sire_features(
                    horse_id=row["horse_id"],
                    race_date=str(row["date"]),
                    distance=row["distance"] or 0,
                    surface=row["surface"] or "",
                    history_df=history_df,
                )
            )

            feature_rows.append(features)

        df = pd.DataFrame(feature_rows)

        # Per-race z-score normalisation
        numeric_cols = [
            c for c in df.columns
            if c not in ["race_id", "entry_id", "target_win", "target_place", "finish_pos", "date", "horse_name"]
            and df[c].dtype in [np.float64, np.float32, np.int64, float, int]
        ]
        df = self.normalise_per_race(df, numeric_cols)

        elapsed = time.time() - t0
        log.info(
            f"✅ Built {len(df)} feature vectors with {len(df.columns)} columns "
            f"in {elapsed:.1f}s ({len(df)/elapsed:.0f} entries/s)"
        )

        # Clean up group indices
        for attr in (
            '_horse_groups', '_jockey_groups', '_trainer_groups', '_course_groups',
            '_jt_combo_groups', '_speed_baselines', '_field_quality_cache', '_weight_field_cache',
            '_bms_name_cache', '_bms_offspring_cache',
        ):
            if hasattr(self, attr):
                delattr(self, attr)

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
