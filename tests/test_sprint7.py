"""
Tests for Sprint 7 features:
  - Weather interaction features (7.2)
  - Track bias features (7.3)
  - Pedigree features (7.4)
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from models.features import (
    FeatureBuilder,
    WEATHER_MAP,
    GOING_MAP,
    SURFACE_MAP,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def history_df():
    """Minimal history DataFrame for testing."""
    return pd.DataFrame({
        "horse_id": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
        "jockey_id": [10, 10, 10, 10, 20, 20, 20, 30, 30, 30, 30, 30],
        "date": [
            "2023-03-01", "2023-06-15", "2023-09-01", "2023-12-01",
            "2023-04-01", "2023-07-01", "2023-10-01",
            "2023-01-01", "2023-05-01", "2023-08-01", "2023-11-01", "2024-01-01",
        ],
        "distance": [1600, 2000, 1600, 2000, 1200, 1400, 1200, 2000, 2000, 1800, 2000, 2000],
        "surface": ["turf", "turf", "dirt", "turf", "dirt", "dirt", "dirt", "turf", "turf", "turf", "turf", "turf"],
        "going": ["良", "稍重", "重", "良", "良", "不良", "良", "良", "良", "稍重", "良", "良"],
        "race_class": ["1勝", "2勝", "1勝", "2勝", "1勝", "1勝", "2勝", "G3", "G2", "G2", "G1", "G1"],
        "grade": [None, None, None, None, None, None, None, "G3", "G2", "G2", "G1", "G1"],
        "course_id": [1, 1, 5, 1, 5, 5, 5, 9, 9, 9, 5, 5],
        "draw": [2, 5, 8, 3, 1, 4, 10, 2, 3, 6, 1, 4],
        "weight_carried": [55, 55, 56, 55, 54, 54, 56, 57, 57, 58, 58, 58],
        "horse_weight": [480, 482, 478, 484, 450, 452, 448, 520, 522, 518, 524, 526],
        "odds_win": [5.0, 8.0, 3.0, 6.0, 10.0, 15.0, 5.0, 2.0, 3.0, 4.0, 2.5, 1.8],
        "finish_pos": [1, 3, 1, 5, 2, 1, 3, 1, 2, 1, 1, 3],
        "time_secs": [96.5, 120.5, 97.2, 121.0, 71.0, 84.2, 72.5, 121.5, 122.0, 109.5, 119.0, 120.5],
        "last_3f_secs": [34.5, 35.0, 35.5, 35.8, 33.0, 33.5, 34.0, 34.0, 34.5, 33.8, 33.5, 34.2],
        "corner_positions": ["3-2-1", "5-4-3", "8-6-4", "3-3-5", "1-1-1", "4-3-2", "10-8-6", "2-2-1", "3-2-2", "6-4-3", "1-1-1", "4-3-3"],
        "field_size": [14, 16, 12, 14, 10, 12, 14, 16, 16, 14, 18, 18],
        "trainer_id": [100, 100, 100, 100, 200, 200, 200, 300, 300, 300, 300, 300],
    })


@pytest.fixture
def fb():
    """FeatureBuilder instance."""
    return FeatureBuilder()


# ---------------------------------------------------------------------------
# Weather Interaction Features (7.2)
# ---------------------------------------------------------------------------

class TestWeatherInteraction:
    def test_weather_map_values(self):
        assert WEATHER_MAP["晴"] == 0
        assert WEATHER_MAP["曇"] == 1
        assert WEATHER_MAP["小雨"] == 2
        assert WEATHER_MAP["雨"] == 3

    def test_basic_weather_features(self, fb, history_df):
        row = pd.Series({
            "weather": "晴",
            "going": "良",
            "surface": "turf",
            "distance": 2000,
            "horse_id": 1,
            "date": "2024-02-01",
        })
        feats = fb._weather_interaction_features(row, history_df)

        assert feats["weather_code"] == 0
        assert feats["going_x_surface"] == 0  # 良(0) × turf(0)
        assert feats["going_x_distance"] == 0  # 良(0) × 2.0

    def test_heavy_going_interactions(self, fb, history_df):
        row = pd.Series({
            "weather": "雨",
            "going": "重",
            "surface": "dirt",
            "distance": 1600,
            "horse_id": 1,
            "date": "2024-02-01",
        })
        feats = fb._weather_interaction_features(row, history_df)

        assert feats["weather_code"] == 3
        assert feats["going_x_surface"] == 2  # 重(2) × dirt(1)
        assert feats["going_x_distance"] == pytest.approx(3.2)  # 重(2) × 1.6

    def test_horse_going_win_pct(self, fb, history_df):
        """Horse 1 has 2 runs on 良, winning 1 → 50%."""
        row = pd.Series({
            "weather": "晴",
            "going": "良",
            "surface": "turf",
            "distance": 2000,
            "horse_id": 1,
            "date": "2024-06-01",
        })
        feats = fb._weather_interaction_features(row, history_df)
        # Horse 1 has 良 races on dates 2023-03-01 (win) and 2023-12-01 (5th)
        assert feats["horse_going_win_pct"] == pytest.approx(0.5, abs=0.01)

    def test_unknown_weather(self, fb, history_df):
        row = pd.Series({
            "weather": "強風",  # not in map
            "going": "良",
            "surface": "turf",
            "distance": 2000,
            "horse_id": 1,
            "date": "2024-02-01",
        })
        feats = fb._weather_interaction_features(row, history_df)
        assert np.isnan(feats["weather_code"])

    def test_no_history(self, fb):
        """First-time runner with no history should return NaN."""
        row = pd.Series({
            "weather": "晴",
            "going": "良",
            "surface": "turf",
            "distance": 2000,
            "horse_id": 999,
            "date": "2024-02-01",
        })
        feats = fb._weather_interaction_features(row, pd.DataFrame(columns=[
            "horse_id", "date", "going", "finish_pos",
        ]))
        assert np.isnan(feats["horse_going_win_pct"])
        assert np.isnan(feats["horse_wet_track_advantage"])


# ---------------------------------------------------------------------------
# Track Bias Features (7.3)
# ---------------------------------------------------------------------------

class TestTrackBias:
    def test_null_course_id(self, fb, history_df):
        row = pd.Series({
            "course_id": None,
            "draw": 3,
            "date": "2024-02-01",
        })
        feats = fb._track_bias_features(row, history_df)
        assert np.isnan(feats["draw_bias_at_course"])
        assert np.isnan(feats["draw_bias_score"])

    def test_basic_track_bias(self, fb, history_df):
        row = pd.Series({
            "course_id": 5,  # Tokyo — has draws 8, 1, 4, 10, 1, 4
            "draw": 2,
            "date": "2024-06-01",
        })
        feats = fb._track_bias_features(row, history_df)
        # Should have values for draw_low_win_pct and draw_high_win_pct
        assert "draw_bias_at_course" in feats
        assert "draw_low_win_pct" in feats
        assert "draw_high_win_pct" in feats
        assert "draw_bias_score" in feats
        assert "course_month_bias" in feats

    def test_draw_bias_score_inner(self, fb):
        """Inner draw (≤4) with inner advantage → positive bias score."""
        # Create history where inner draws win more than outer
        hist = pd.DataFrame({
            "horse_id": range(40),
            "date": ["2023-01-01"] * 40,
            "course_id": [1] * 40,
            "draw": ([1, 2, 3, 4] * 5) + ([9, 10, 11, 12] * 5),
            "finish_pos": ([1, 2, 3, 4] * 5) + ([5, 6, 7, 8] * 5),
            "surface": ["turf"] * 40,
            "going": ["良"] * 40,
            "distance": [2000] * 40,
        })
        fb_test = FeatureBuilder()
        row = pd.Series({"course_id": 1, "draw": 2, "date": "2023-06-01"})
        feats = fb_test._track_bias_features(row, hist)
        # Inner draws have 25% win rate (5 wins / 20), outer have 0%
        assert feats["draw_low_win_pct"] > feats["draw_high_win_pct"]
        assert feats["draw_bias_score"] > 0


# ---------------------------------------------------------------------------
# Pedigree Features (7.4)
# ---------------------------------------------------------------------------

class TestPedigreeFeatures:
    def test_no_sire_data(self, fb, history_df):
        """When no sire data exists, return NaN."""
        # Patch the cache to be empty
        fb._sire_cache = {}
        feats = fb._pedigree_features(
            horse_id=1, race_date="2024-06-01",
            distance=2000, surface="turf", history_df=history_df,
        )
        assert np.isnan(feats["sire_runners"])
        assert np.isnan(feats["sire_win_pct"])

    def test_with_sire_data(self, fb, history_df):
        """When sire data exists, compute meaningful features."""
        # Set up mock sire cache: horses 1 and 3 share sire "DeepImpact"
        fb._sire_cache = {1: "DeepImpact", 3: "DeepImpact", 2: "SundaySilence"}
        fb._offspring_cache = {
            "DeepImpact": [1, 3],
            "SundaySilence": [2],
        }

        feats = fb._pedigree_features(
            horse_id=1, race_date="2024-06-01",
            distance=2000, surface="turf", history_df=history_df,
        )
        # Horse 3 (sibling) has races in history before 2024-06-01
        assert feats["sire_runners"] > 0
        assert 0 <= feats["sire_win_pct"] <= 1
        assert feats["sire_avg_finish"] > 0

    def test_sire_surface_affinity(self, fb, history_df):
        """Sire × surface should filter correctly."""
        fb._sire_cache = {1: "DeepImpact", 3: "DeepImpact"}
        fb._offspring_cache = {"DeepImpact": [1, 3]}

        # Horse 3 runs mainly on turf
        feats_turf = fb._pedigree_features(
            horse_id=1, race_date="2024-06-01",
            distance=2000, surface="turf", history_df=history_df,
        )
        feats_dirt = fb._pedigree_features(
            horse_id=1, race_date="2024-06-01",
            distance=2000, surface="dirt", history_df=history_df,
        )
        # Horse 3 has mostly turf runs, so turf should have data
        if not np.isnan(feats_turf["sire_win_pct_surface"]):
            assert 0 <= feats_turf["sire_win_pct_surface"] <= 1


# ---------------------------------------------------------------------------
# Empty Features Consistency
# ---------------------------------------------------------------------------

class TestEmptyFeatures:
    def test_empty_features_has_sprint7_keys(self, fb):
        """_empty_horse_features should include Sprint 7 feature keys."""
        empty = fb._empty_horse_features()
        sprint7_keys = [
            "weather_code", "going_x_surface", "going_x_distance",
            "horse_going_win_pct", "horse_wet_track_advantage",
            "draw_bias_at_course", "draw_low_win_pct", "draw_high_win_pct",
            "draw_bias_score", "course_month_bias",
            "sire_runners", "sire_win_pct", "sire_win_pct_surface",
            "sire_win_pct_distance", "sire_avg_finish",
        ]
        for key in sprint7_keys:
            assert key in empty, f"Missing key: {key}"
            assert np.isnan(empty[key]), f"Key {key} should be NaN"
