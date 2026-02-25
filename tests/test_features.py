"""
Unit tests for the feature engineering pipeline.
Tests feature computation, normalisation, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from models.features import (
    CLASS_RANK,
    GOING_MAP,
    SURFACE_MAP,
    SEX_MAP,
    FeatureBuilder,
)


# ---------------------------------------------------------------------------
# Mapping Tests
# ---------------------------------------------------------------------------

class TestMappings:
    def test_class_rank_g1_is_strongest(self):
        assert CLASS_RANK["G1"] == 1
        assert CLASS_RANK["GI"] == 1

    def test_class_rank_maiden_is_weakest(self):
        assert CLASS_RANK["新馬"] == 10

    def test_class_rank_ordering(self):
        """G1 < G2 < G3 < L < OP < ... < 新馬"""
        assert CLASS_RANK["G1"] < CLASS_RANK["G2"] < CLASS_RANK["G3"]
        assert CLASS_RANK["G3"] < CLASS_RANK["L"] < CLASS_RANK["OP"]
        assert CLASS_RANK["OP"] < CLASS_RANK["未勝利"] < CLASS_RANK["新馬"]

    def test_going_map_values(self):
        assert GOING_MAP["良"] == 0  # Good
        assert GOING_MAP["不良"] == 3  # Bad

    def test_surface_map(self):
        assert SURFACE_MAP["turf"] == 0
        assert SURFACE_MAP["dirt"] == 1

    def test_sex_map(self):
        assert SEX_MAP["牡"] == 0  # male
        assert SEX_MAP["牝"] == 1  # female
        assert SEX_MAP["セ"] == 2  # gelding


# ---------------------------------------------------------------------------
# Static Features
# ---------------------------------------------------------------------------

class TestStaticFeatures:
    def setup_method(self):
        self.fb = FeatureBuilder()

    def test_basic_static_features(self):
        row = pd.Series({
            "draw": 3,
            "post_position": 5,
            "weight_carried": 57.0,
            "horse_weight": 480,
            "horse_weight_change": 4,
            "odds_win": 5.0,
            "popularity": 2,
            "distance": 2000,
            "surface": "turf",
            "going": "良",
            "field_size": 16,
            "race_class": "G1",
            "grade": "G1",
            "sex": "牡",
            "birth_year": 2019,
            "date": "2024-06-01",
        })
        features = self.fb._static_features(row)

        assert features["draw"] == 3
        assert features["weight_carried"] == 57.0
        assert features["horse_weight"] == 480
        assert features["horse_weight_change"] == 4
        assert features["odds_win"] == 5.0
        assert features["log_odds"] == pytest.approx(np.log(5.0), rel=1e-5)
        assert features["surface_code"] == 0  # turf
        assert features["going_code"] == 0  # 良 = good
        assert features["class_rank"] == 1  # G1
        assert features["sex_code"] == 0  # 牡 = male
        assert features["age"] == 2024 - 2019

    def test_dirt_surface(self):
        row = pd.Series({"surface": "dirt", "draw": None, "post_position": None,
                         "weight_carried": None, "horse_weight": None, "horse_weight_change": None,
                         "odds_win": None, "popularity": None, "distance": None, "going": None,
                         "field_size": None, "race_class": None, "grade": None, "sex": None,
                         "birth_year": None, "date": None})
        features = self.fb._static_features(row)
        assert features["surface_code"] == 1

    def test_null_odds_produces_nan_log(self):
        row = pd.Series({"odds_win": None, "draw": None, "post_position": None,
                         "weight_carried": None, "horse_weight": None, "horse_weight_change": None,
                         "popularity": None, "distance": None, "surface": None, "going": None,
                         "field_size": None, "race_class": None, "grade": None, "sex": None,
                         "birth_year": None, "date": None})
        features = self.fb._static_features(row)
        assert np.isnan(features["log_odds"])


# ---------------------------------------------------------------------------
# Rolling Horse Features
# ---------------------------------------------------------------------------

class TestHorseRollingFeatures:
    def setup_method(self):
        self.fb = FeatureBuilder()

    def _make_history(self, n_races=5):
        """Create synthetic horse history."""
        return pd.DataFrame({
            "horse_id": [1] * n_races,
            "jockey_id": [10] * n_races,
            "date": pd.date_range("2024-01-01", periods=n_races, freq="14D").astype(str),
            "distance": [2000] * n_races,
            "surface": ["turf"] * n_races,
            "going": ["良"] * n_races,
            "race_class": ["G2"] * n_races,
            "grade": ["G2"] * n_races,
            "course_id": [5] * n_races,
            "draw": list(range(1, n_races + 1)),
            "weight_carried": [57.0] * n_races,
            "horse_weight": [480] * n_races,
            "odds_win": [5.0] * n_races,
            "finish_pos": [1, 3, 2, 5, 1],
            "time_secs": [120.5, 121.0, 120.8, 122.0, 120.3],
            "last_3f_secs": [34.0, 34.5, 33.8, 35.0, 33.5],
            "corner_positions": ["2-2-1-1", "5-4-3-3", "3-3-2-2", "8-7-5-5", "1-1-1-1"],
            "field_size": [16] * n_races,
        })

    def test_first_time_runner_returns_nan(self):
        history = pd.DataFrame(columns=[
            "horse_id", "date", "distance", "surface", "going",
            "race_class", "grade", "course_id", "draw", "weight_carried",
            "horse_weight", "odds_win", "finish_pos", "time_secs",
            "last_3f_secs", "corner_positions", "field_size", "jockey_id",
        ])
        features = self.fb._horse_rolling_features(
            horse_id=999, race_date="2024-07-01", distance=2000,
            surface="turf", course_id=5, history_df=history,
        )
        assert all(np.isnan(v) for v in features.values())

    def test_career_win_pct(self):
        history = self._make_history()
        features = self.fb._horse_rolling_features(
            horse_id=1, race_date="2024-04-01",  # after all 5 races (14-day intervals from Jan 1)
            distance=2000, surface="turf", course_id=5,
            history_df=history,
        )
        # 5 races with finishes [1, 3, 2, 5, 1] → 2 wins out of 5 = 0.40
        assert features["career_win_pct"] == pytest.approx(0.4, rel=0.01)

    def test_empty_horse_features_keys(self):
        empty = self.fb._empty_horse_features()
        expected_keys = {
            "last3_win_pct", "last3_place_pct", "last3_avg_finish", "last3_avg_beaten_pct",
            "last5_win_pct", "career_runs", "career_wins", "career_win_pct",
            "avg_first_corner", "weight_trend",
        }
        assert expected_keys.issubset(set(empty.keys()))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_per_race_zscore(self):
        """Z-score within a race should have mean ≈ 0, std ≈ 1."""
        df = pd.DataFrame({
            "race_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "speed": [10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0],
        })
        result = FeatureBuilder.normalise_per_race(df.copy(), ["speed"])

        # Check z-scores within race 1
        race1 = result[result["race_id"] == 1]
        assert race1["speed_z"].mean() == pytest.approx(0.0, abs=1e-10)
        assert race1["speed_z"].std() == pytest.approx(1.0, abs=0.1)

    def test_zscore_skips_non_numeric(self):
        df = pd.DataFrame({
            "race_id": [1, 1],
            "name": ["Horse A", "Horse B"],
        })
        result = FeatureBuilder.normalise_per_race(df.copy(), ["name"])
        assert "name_z" not in result.columns


# ---------------------------------------------------------------------------
# Jockey Features
# ---------------------------------------------------------------------------

class TestJockeyFeatures:
    def setup_method(self):
        self.fb = FeatureBuilder()

    def test_null_jockey(self):
        features = self.fb._jockey_features(None, "2024-01-01", pd.DataFrame())
        assert np.isnan(features["jockey_win_pct_5"])
        assert np.isnan(features["jockey_win_pct_10"])

    def test_jockey_with_history(self):
        history = pd.DataFrame({
            "jockey_id": [10] * 10,
            "date": pd.date_range("2024-01-01", periods=10, freq="7D").astype(str),
            "finish_pos": [1, 2, 1, 3, 5, 1, 4, 2, 1, 6],
            "horse_id": list(range(10)),
            "odds_win": [5.0, 10.0, 3.0, 8.0, 15.0, 4.0, 20.0, 6.0, 2.5, 30.0],
        })
        features = self.fb._jockey_features(10, "2024-04-01", history)
        assert features["jockey_win_pct_5"] > 0
        assert features["jockey_place_pct_5"] > 0
        assert features["jockey_win_pct_10"] > 0
        assert features["jockey_place_pct_10"] > 0
        assert "jockey_roi_10" in features
        assert features["jockey_recent_wins"] > 0

    def test_jockey_roi_calculation(self):
        """ROI should be sum(odds_win for wins) / N - 1."""
        history = pd.DataFrame({
            "jockey_id": [10] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="7D").astype(str),
            "finish_pos": [1, 2, 3, 4, 5],
            "horse_id": list(range(5)),
            "odds_win": [5.0, 10.0, 3.0, 8.0, 15.0],
        })
        features = self.fb._jockey_features(10, "2024-04-01", history)
        # 1 win (odds=5.0) out of 5 races → ROI = 5.0/5 - 1 = 0.0
        assert features["jockey_roi_10"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Trainer Features
# ---------------------------------------------------------------------------

class TestTrainerFeatures:
    def setup_method(self):
        self.fb = FeatureBuilder()

    def test_null_trainer(self):
        features = self.fb._trainer_features(None, "2024-01-01", pd.DataFrame())
        assert np.isnan(features["trainer_win_pct_10"])
        assert np.isnan(features["trainer_place_pct_10"])
        assert np.isnan(features["trainer_roi_10"])

    def test_trainer_with_history(self):
        history = pd.DataFrame({
            "trainer_id": [20] * 10,
            "date": pd.date_range("2024-01-01", periods=10, freq="7D").astype(str),
            "finish_pos": [1, 3, 2, 5, 1, 8, 4, 1, 6, 2],
            "odds_win": [3.0, 6.0, 5.0, 12.0, 4.0, 20.0, 8.0, 2.5, 15.0, 7.0],
        })
        features = self.fb._trainer_features(20, "2024-04-01", history)
        assert features["trainer_win_pct_10"] > 0
        assert features["trainer_place_pct_10"] > 0
        assert "trainer_roi_10" in features

    def test_trainer_no_history_column(self):
        """No trainer_id column in history should return NaN."""
        history = pd.DataFrame({
            "date": ["2024-01-01"],
            "finish_pos": [1],
        })
        features = self.fb._trainer_features(20, "2024-04-01", history)
        assert np.isnan(features["trainer_win_pct_10"])


# ---------------------------------------------------------------------------
# Pace Features
# ---------------------------------------------------------------------------

class TestPaceFeatures:
    def setup_method(self):
        self.fb = FeatureBuilder()

    def test_pace_features_structure(self):
        """Pace features should return the expected keys."""
        # Test with empty race_df (should return null features)
        race_df = pd.DataFrame({"race_id": [], "horse_id": []})
        history_df = pd.DataFrame()

        features = self.fb._get_pace_features(999, 1, race_df, history_df)

        assert "pace_win_prob" in features
        assert "pace_place_prob" in features
        assert "pace_style_front" in features
        assert "pace_style_stalk" in features
        assert "pace_style_closer" in features
        assert "pace_style_deep" in features

