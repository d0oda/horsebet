"""
Tests for Sprint 8 features:
  - Speed figures (8.1)
  - Jockey-trainer combo (8.2)
  - Field quality (8.3)
  - Beaten lengths (8.4)
  - Fitness curve (8.5)
  - Age × class (8.6)
  - Weight vs field (8.7)
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from models.features import (
    FeatureBuilder,
    CLASS_RANK,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def history_df():
    """Minimal history DataFrame with margin column for Sprint 8 tests."""
    return pd.DataFrame({
        "horse_id": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "jockey_id": [10, 10, 11, 10, 10, 11, 11, 10, 10, 10],
        "trainer_id": [100, 100, 100, 100, 200, 200, 200, 100, 100, 100],
        "date": [
            "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01",
            "2025-01-15", "2025-02-15", "2025-03-15",
            "2025-01-10", "2025-02-10", "2025-03-10",
        ],
        "distance": [2000, 1600, 2000, 1800, 2000, 2000, 1600, 1200, 1200, 1400],
        "surface": ["turf"] * 10,
        "going": ["良", "良", "重", "良", "良", "重", "良", "良", "良", "稍重"],
        "race_class": ["OP", "G3", "G1", "G2", "OP", "OP", "未勝利", "新馬", "新馬", "未勝利"],
        "grade": [None] * 10,
        "course_id": [1, 1, 2, 1, 1, 2, 1, 3, 3, 3],
        "draw": [3, 5, 2, 8, 1, 4, 7, 2, 6, 4],
        "weight_carried": [57, 55, 56, 57, 54, 55, 53, 54, 54, 55],
        "horse_weight": [480, 478, 482, 480, 460, 462, 458, 440, 442, 444],
        "odds_win": [5.0, 8.0, 3.0, 12.0, 15.0, 20.0, 50.0, 10.0, 12.0, 8.0],
        "finish_pos": [1, 3, 2, 5, 1, 4, 8, 2, 1, 3],
        "time_secs": [120.5, 96.2, 121.0, 108.5, 121.5, 122.0, 97.0, 71.0, 70.5, 85.0],
        "last_3f_secs": [35.0, 34.5, 35.5, 34.8, 36.0, 35.8, 36.5, 34.0, 33.5, 34.5],
        "corner_positions": ["3-3-2-1", "5-4-3-2", "2-2-1-1", "8-7-6-5",
                             "1-1-1-1", "4-5-4-3", "7-8-8-8",
                             "2-2-1-2", "6-5-4-1", "4-3-3-3"],
        "margin": [0.0, 0.5, 0.2, 3.0, 0.0, 1.5, 8.0, 0.3, 0.0, 1.0],
        "field_size": [16, 14, 12, 16, 16, 14, 18, 10, 10, 12],
    })


@pytest.fixture
def fb():
    """FeatureBuilder instance."""
    return FeatureBuilder()


# ---------------------------------------------------------------------------
# Speed Figures (8.1)
# ---------------------------------------------------------------------------

class TestSpeedFigures:
    def test_no_history_returns_nan(self, fb, history_df):
        """First-time runner should return NaN speed figures."""
        fb._horse_groups = dict(list(history_df.groupby("horse_id")))
        result = fb._speed_figure_features(
            horse_id=999, race_date="2025-05-01",
            distance=2000, course_id=1, going="良",
            history_df=history_df,
        )
        assert np.isnan(result["speed_figure_last"])
        assert np.isnan(result["speed_figure_best"])
        assert np.isnan(result["speed_figure_avg3"])

    @patch.object(FeatureBuilder, "_get_speed_baseline_pit", return_value=120.0)
    def test_speed_figure_calculation(self, mock_baseline, fb, history_df):
        """Speed figures should be (baseline - actual) / baseline * 1000."""
        fb._horse_groups = dict(list(history_df.groupby("horse_id")))
        result = fb._speed_figure_features(
            horse_id=1, race_date="2025-05-01",
            distance=2000, course_id=1, going="良",
            history_df=history_df,
        )
        # Should have non-NaN figures
        assert not np.isnan(result["speed_figure_last"])
        assert not np.isnan(result["speed_figure_best"])
        assert not np.isnan(result["speed_figure_avg3"])
        # Best should be >= last (or equal if only one matching race)
        assert result["speed_figure_best"] >= result["speed_figure_last"]

    def test_empty_baselines(self, fb):
        """Empty history should produce empty baselines and NaN features."""
        empty_hist = pd.DataFrame(columns=[
            "horse_id", "time_secs", "course_id", "distance",
            "going", "date", "finish_pos",
        ])
        fb._horse_groups = {}
        result = fb._speed_figure_features(
            horse_id=1, race_date="2025-05-01",
            distance=2000, course_id=1, going="良",
            history_df=empty_hist,
        )
        for v in result.values():
            assert np.isnan(v)


# ---------------------------------------------------------------------------
# Jockey-Trainer Combo (8.2)
# ---------------------------------------------------------------------------

class TestJockeyTrainerCombo:
    def test_null_jockey_or_trainer(self, fb, history_df):
        """Null jockey or trainer should return NaN."""
        result = fb._jockey_trainer_combo_features(
            jockey_id=None, trainer_id=100,
            race_date="2025-05-01", history_df=history_df,
        )
        assert np.isnan(result["jt_combo_runs"])

        result2 = fb._jockey_trainer_combo_features(
            jockey_id=10, trainer_id=None,
            race_date="2025-05-01", history_df=history_df,
        )
        assert np.isnan(result2["jt_combo_win_pct"])

    def test_combo_with_history(self, fb, history_df):
        """Jockey 10 + Trainer 100 should have measurable stats."""
        result = fb._jockey_trainer_combo_features(
            jockey_id=10, trainer_id=100,
            race_date="2025-05-01", history_df=history_df,
        )
        # Jockey 10 with Trainer 100: horse 1 rows (dates 01-01, 02-01, 04-01)
        # + horse 3 rows (dates 01-10, 02-10, 03-10)
        assert result["jt_combo_runs"] > 0
        assert 0 <= result["jt_combo_win_pct"] <= 1
        assert 0 <= result["jt_combo_place_pct"] <= 1

    def test_no_shared_history(self, fb, history_df):
        """A jockey-trainer pair with no shared history → NaN."""
        result = fb._jockey_trainer_combo_features(
            jockey_id=99, trainer_id=999,  # not in test data at all
            race_date="2025-05-01", history_df=history_df,
        )
        assert np.isnan(result["jt_combo_runs"])


# ---------------------------------------------------------------------------
# Beaten Lengths (8.4)
# ---------------------------------------------------------------------------

class TestBeatenLengths:
    def test_beaten_lengths_computed(self, fb, history_df):
        """Horse 1 has margins [0.0, 0.5, 0.2, 3.0] — averages should be populated."""
        fb._horse_groups = dict(list(history_df.groupby("horse_id")))
        result = fb._horse_rolling_features(
            horse_id=1, race_date="2025-05-01",
            distance=2000, surface="turf", course_id=1,
            history_df=history_df,
        )
        assert not np.isnan(result["beaten_lengths_avg3"])
        assert not np.isnan(result["beaten_lengths_best"])
        assert not np.isnan(result["class_adjusted_margin"])
        # Best margin should be 0.0 (the win)
        assert result["beaten_lengths_best"] == 0.0

    def test_no_margin_column(self, fb):
        """History without margin column should return NaN."""
        hist_no_margin = pd.DataFrame({
            "horse_id": [1, 1],
            "date": ["2025-01-01", "2025-02-01"],
            "distance": [2000, 2000],
            "surface": ["turf", "turf"],
            "going": ["良", "良"],
            "race_class": ["OP", "OP"],
            "grade": [None, None],
            "course_id": [1, 1],
            "draw": [3, 5],
            "weight_carried": [57, 55],
            "horse_weight": [480, 478],
            "odds_win": [5.0, 8.0],
            "finish_pos": [1, 3],
            "time_secs": [120.5, 121.0],
            "last_3f_secs": [35.0, 34.5],
            "corner_positions": ["3-3-2-1", "5-4-3-2"],
            "field_size": [16, 14],
        })
        fb._horse_groups = dict(list(hist_no_margin.groupby("horse_id")))
        result = fb._horse_rolling_features(
            horse_id=1, race_date="2025-05-01",
            distance=2000, surface="turf", course_id=1,
            history_df=hist_no_margin,
        )
        assert np.isnan(result["beaten_lengths_avg3"])


# ---------------------------------------------------------------------------
# Fitness Curve (8.5)
# ---------------------------------------------------------------------------

class TestFitnessCurve:
    def test_fresh_horse(self, fb, history_df):
        """Horse racing 21 days after last run should be 'fresh'."""
        fb._horse_groups = dict(list(history_df.groupby("horse_id")))
        # Horse 1's last race is 2025-04-01, race date 2025-04-22 = 21 days
        result = fb._horse_rolling_features(
            horse_id=1, race_date="2025-04-22",
            distance=2000, surface="turf", course_id=1,
            history_df=history_df,
        )
        assert result["is_fresh"] == 1
        assert result["is_rested"] == 0
        assert result["is_stale"] == 0

    def test_rested_horse(self, fb, history_df):
        """Horse racing 45 days after last run should be 'rested'."""
        fb._horse_groups = dict(list(history_df.groupby("horse_id")))
        # Horse 1's last race is 2025-04-01, race date 2025-05-16 = 45 days
        result = fb._horse_rolling_features(
            horse_id=1, race_date="2025-05-16",
            distance=2000, surface="turf", course_id=1,
            history_df=history_df,
        )
        assert result["is_fresh"] == 0
        assert result["is_rested"] == 1
        assert result["is_stale"] == 0

    def test_stale_horse(self, fb, history_df):
        """Horse racing 120 days after last run should be 'stale'."""
        fb._horse_groups = dict(list(history_df.groupby("horse_id")))
        # Horse 1's last race is 2025-04-01, race date 2025-07-30 = 120 days
        result = fb._horse_rolling_features(
            horse_id=1, race_date="2025-07-30",
            distance=2000, surface="turf", course_id=1,
            history_df=history_df,
        )
        assert result["is_fresh"] == 0
        assert result["is_rested"] == 0
        assert result["is_stale"] == 1

    def test_first_time_runner_nan(self, fb, history_df):
        """First-time runner should have NaN fitness flags."""
        fb._horse_groups = dict(list(history_df.groupby("horse_id")))
        result = fb._horse_rolling_features(
            horse_id=999, race_date="2025-05-01",
            distance=2000, surface="turf", course_id=1,
            history_df=history_df,
        )
        assert np.isnan(result["is_fresh"])


# ---------------------------------------------------------------------------
# Field Quality (8.3)
# ---------------------------------------------------------------------------

class TestFieldQuality:
    def test_field_quality_cached(self, fb, history_df):
        """Field quality should be computed and cached per race."""
        fb._horse_groups = dict(list(history_df.groupby("horse_id")))
        race_df = pd.DataFrame({
            "race_id": [1, 1, 1],
            "horse_id": [1, 2, 3],
        })
        result = fb._field_quality_features(
            race_id=1, horse_career_win_pct=0.25,
            race_df=race_df, history_df=history_df,
        )
        assert not np.isnan(result["field_avg_career_win_pct"])
        assert not np.isnan(result["horse_vs_field_quality"])
        # Should be cached now
        assert 1 in fb._field_quality_cache

    def test_field_quality_nan_career(self, fb, history_df):
        """NaN career_win_pct should still populate field avg but NaN the diff."""
        fb._horse_groups = dict(list(history_df.groupby("horse_id")))
        race_df = pd.DataFrame({
            "race_id": [1, 1],
            "horse_id": [1, 2],
        })
        result = fb._field_quality_features(
            race_id=1, horse_career_win_pct=np.nan,
            race_df=race_df, history_df=history_df,
        )
        assert not np.isnan(result["field_avg_career_win_pct"])
        assert np.isnan(result["horse_vs_field_quality"])


# ---------------------------------------------------------------------------
# Empty Features Consistency
# ---------------------------------------------------------------------------

class TestEmptyFeaturesSprint8:
    def test_empty_features_has_sprint8_keys(self, fb):
        """_empty_horse_features should include all Sprint 8 feature keys."""
        empty = fb._empty_horse_features()
        sprint8_keys = [
            "beaten_lengths_avg3", "beaten_lengths_best", "class_adjusted_margin",
            "is_fresh", "is_rested", "is_stale",
            "speed_figure_last", "speed_figure_best", "speed_figure_avg3",
            "jt_combo_runs", "jt_combo_win_pct", "jt_combo_place_pct",
            "field_avg_career_win_pct", "horse_vs_field_quality",
            "weight_vs_field_avg", "weight_per_kg_body",
            "age_x_class", "is_improving_3yo",
        ]
        for key in sprint8_keys:
            assert key in empty, f"Missing key in _empty_horse_features: {key}"
            assert np.isnan(empty[key]), f"Key {key} should be NaN for first-time runner"
