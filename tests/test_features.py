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


# ---------------------------------------------------------------------------
# Class Change Features (Research Backlog #1)
# ---------------------------------------------------------------------------

class TestClassChangeFeatures:
    def setup_method(self):
        self.fb = FeatureBuilder()

    def _make_history_with_class(self):
        """Create history with varying class levels."""
        return pd.DataFrame({
            "horse_id": [1] * 5,
            "jockey_id": [10] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="14D").astype(str),
            "distance": [2000] * 5,
            "surface": ["turf"] * 5,
            "going": ["良"] * 5,
            "race_class": ["G1", "G2", "G3", "OP", "3勝"],  # newest first after sort
            "grade": ["G1", "G2", "G3", None, None],
            "course_id": [5] * 5,
            "draw": [3] * 5,
            "weight_carried": [57.0] * 5,
            "horse_weight": [480] * 5,
            "odds_win": [5.0] * 5,
            "finish_pos": [1, 3, 2, 5, 1],
            "time_secs": [120.5, 121.0, 120.8, 122.0, 120.3],
            "last_3f_secs": [34.0, 34.5, 33.8, 35.0, 33.5],
            "corner_positions": ["2-2-1-1", "5-4-3-3", "3-3-2-2", "8-7-5-5", "1-1-1-1"],
            "field_size": [16] * 5,
        })

    def test_class_change_computed(self):
        """Horse with class history should have class_change feature."""
        history = self._make_history_with_class()
        features = self.fb._horse_rolling_features(
            horse_id=1, race_date="2024-04-01",
            distance=2000, surface="turf", course_id=5,
            history_df=history,
        )
        assert "class_change" in features
        assert "class_drops_last5" in features
        assert "class_rises_last5" in features
        assert "class_at_last_win" in features

    def test_class_at_last_win(self):
        """class_at_last_win should reflect the class rank of the most recent win."""
        history = self._make_history_with_class()
        features = self.fb._horse_rolling_features(
            horse_id=1, race_date="2024-04-01",
            distance=2000, surface="turf", course_id=5,
            history_df=history,
        )
        # First-time runner has NaN
        assert not np.isnan(features["class_at_last_win"])

    def test_first_time_runner_class_nan(self):
        """First-time runner should have NaN class features."""
        empty = self.fb._empty_horse_features()
        assert np.isnan(empty["class_change"])
        assert np.isnan(empty["class_drops_last5"])
        assert np.isnan(empty["class_rises_last5"])
        assert np.isnan(empty["class_at_last_win"])


# ---------------------------------------------------------------------------
# Trainer 14-Day Form (Research Backlog #2)
# ---------------------------------------------------------------------------

class TestTrainer14DayForm:
    def setup_method(self):
        self.fb = FeatureBuilder()

    def test_trainer_14d_with_recent_activity(self):
        """Trainer with runs in last 14 days should have non-NaN features."""
        history = pd.DataFrame({
            "trainer_id": [20] * 10,
            "date": pd.date_range("2024-03-18", periods=10, freq="2D").astype(str),
            "finish_pos": [1, 3, 2, 5, 1, 8, 4, 1, 6, 2],
            "odds_win": [3.0, 6.0, 5.0, 12.0, 4.0, 20.0, 8.0, 2.5, 15.0, 7.0],
        })
        features = self.fb._trainer_features(20, "2024-04-08", history)
        assert features["trainer_14d_runs"] > 0
        assert not np.isnan(features["trainer_14d_win_pct"])
        assert not np.isnan(features["trainer_14d_place_pct"])

    def test_trainer_14d_no_recent_activity(self):
        """Trainer with no runs in last 14 days should have 0 runs and NaN rates."""
        history = pd.DataFrame({
            "trainer_id": [20] * 3,
            "date": ["2024-01-01", "2024-01-15", "2024-02-01"],
            "finish_pos": [1, 3, 2],
            "odds_win": [3.0, 6.0, 5.0],
        })
        features = self.fb._trainer_features(20, "2024-04-01", history)
        assert features["trainer_14d_runs"] == 0
        assert np.isnan(features["trainer_14d_win_pct"])


# ---------------------------------------------------------------------------
# Course × Jockey Features (Research Backlog #3)
# ---------------------------------------------------------------------------

class TestCourseJockeyFeatures:
    def setup_method(self):
        self.fb = FeatureBuilder()

    def test_jockey_with_course_history(self):
        """Jockey with course-specific history should have computed stats."""
        history = pd.DataFrame({
            "jockey_id": [10] * 10,
            "course_id": [5, 5, 5, 5, 5, 8, 8, 8, 8, 8],
            "date": pd.date_range("2024-01-01", periods=10, freq="7D").astype(str),
            "finish_pos": [1, 2, 1, 3, 5, 1, 4, 2, 1, 6],
        })
        features = self.fb._course_jockey_features(10, 5, "2024-04-01", history)
        assert features["jockey_course_runs"] == 5
        assert features["jockey_course_win_pct"] == pytest.approx(0.4, rel=0.01)  # 2/5

    def test_null_jockey(self):
        """Null jockey should return NaN features."""
        features = self.fb._course_jockey_features(None, 5, "2024-04-01", pd.DataFrame())
        assert np.isnan(features["jockey_course_runs"])
        assert np.isnan(features["jockey_course_win_pct"])
        assert np.isnan(features["jockey_course_place_pct"])

    def test_jockey_at_new_course(self):
        """Jockey with no runs at this course should return NaN."""
        history = pd.DataFrame({
            "jockey_id": [10] * 5,
            "course_id": [8] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="7D").astype(str),
            "finish_pos": [1, 2, 3, 4, 5],
        })
        features = self.fb._course_jockey_features(10, 5, "2024-04-01", history)
        assert np.isnan(features["jockey_course_runs"])


# ---------------------------------------------------------------------------
# Seasonal Form Features (Sprint 9.1)
# ---------------------------------------------------------------------------

class TestSeasonalFormFeatures:
    def setup_method(self):
        self.fb = FeatureBuilder()

    def test_month_extraction(self):
        """Should correctly extract month_of_year from race date."""
        # Need _horse_groups for the method — provide empty
        self.fb._horse_groups = {}
        features = self.fb._seasonal_form_features(
            horse_id=999, race_date="2024-06-15",
            history_df=pd.DataFrame(),
        )
        assert features["month_of_year"] == 6

    def test_season_codes(self):
        """Spring=0, Summer=1, Autumn=2, Winter=3."""
        self.fb._horse_groups = {}
        # Spring (March)
        f = self.fb._seasonal_form_features(999, "2024-03-01", pd.DataFrame())
        assert f["season_code"] == 0
        # Summer (July)
        f = self.fb._seasonal_form_features(999, "2024-07-01", pd.DataFrame())
        assert f["season_code"] == 1
        # Autumn (October)
        f = self.fb._seasonal_form_features(999, "2024-10-01", pd.DataFrame())
        assert f["season_code"] == 2
        # Winter (January)
        f = self.fb._seasonal_form_features(999, "2024-01-01", pd.DataFrame())
        assert f["season_code"] == 3

    def test_horse_month_win_pct_with_history(self):
        """Horse with enough same-month races should get a win pct."""
        history = pd.DataFrame({
            "horse_id": [1] * 4,
            "date": ["2023-06-01", "2023-06-15", "2022-06-10", "2024-01-10"],
            "finish_pos": [1, 3, 1, 2],
        })
        self.fb._horse_groups = {1: history}
        features = self.fb._seasonal_form_features(
            horse_id=1, race_date="2024-06-20",
            history_df=history,
        )
        # 3 races in June (before 2024-06-20), 2 wins → 2/3 ≈ 0.667
        assert features["horse_month_win_pct"] == pytest.approx(2/3, rel=0.01)

    def test_horse_month_win_pct_insufficient_data(self):
        """Horse with < 2 same-month races should return NaN."""
        history = pd.DataFrame({
            "horse_id": [1],
            "date": ["2023-06-01"],
            "finish_pos": [1],
        })
        self.fb._horse_groups = {1: history}
        features = self.fb._seasonal_form_features(
            horse_id=1, race_date="2024-06-20",
            history_df=history,
        )
        assert np.isnan(features["horse_month_win_pct"])

    def test_no_history(self):
        """Horse with no history should return NaN for monthly win pct."""
        self.fb._horse_groups = {}
        features = self.fb._seasonal_form_features(
            horse_id=999, race_date="2024-06-20",
            history_df=pd.DataFrame(),
        )
        assert np.isnan(features["horse_month_win_pct"])
        assert features["month_of_year"] == 6


# ---------------------------------------------------------------------------
# Broodmare Sire Features (Sprint 9.2)
# ---------------------------------------------------------------------------

class TestBroodmareSireFeatures:
    def setup_method(self):
        self.fb = FeatureBuilder()
        # Set empty BMS caches to avoid DB calls in tests
        self.fb._bms_name_cache = {}
        self.fb._bms_offspring_cache = {}

    def test_null_broodmare_sire(self):
        """Horse with no broodmare sire should return NaN features."""
        features = self.fb._broodmare_sire_features(
            horse_id=999, race_date="2024-06-20",
            distance=2000, surface="turf",
            history_df=pd.DataFrame(),
        )
        assert np.isnan(features["bms_runners"])
        assert np.isnan(features["bms_win_pct"])
        assert np.isnan(features["bms_win_pct_surface"])
        assert np.isnan(features["bms_win_pct_distance"])

    def test_broodmare_sire_with_offspring(self):
        """BMS with offspring in history should compute stats."""
        # Set up caches directly
        self.fb._bms_name_cache = {1: "DeepImpact"}
        self.fb._bms_offspring_cache = {"DeepImpact": [2, 3, 4]}

        history = pd.DataFrame({
            "horse_id": [2, 2, 3, 3, 4],
            "date": ["2024-01-01", "2024-02-01", "2024-01-15", "2024-03-01", "2024-02-15"],
            "finish_pos": [1, 3, 2, 1, 5],
            "distance": [2000, 2000, 1800, 2200, 2000],
            "surface": ["turf", "turf", "turf", "dirt", "turf"],
        })

        features = self.fb._broodmare_sire_features(
            horse_id=1, race_date="2024-06-01",
            distance=2000, surface="turf",
            history_df=history,
        )
        assert features["bms_runners"] == 5
        assert features["bms_win_pct"] == pytest.approx(2/5, rel=0.01)

    def test_broodmare_sire_feature_keys(self):
        """Should always return the expected feature keys."""
        features = self.fb._broodmare_sire_features(
            horse_id=999, race_date="2024-06-20",
            distance=2000, surface="turf",
            history_df=pd.DataFrame(),
        )
        expected_keys = {"bms_runners", "bms_win_pct", "bms_win_pct_surface", "bms_win_pct_distance"}
        assert expected_keys == set(features.keys())


# ---------------------------------------------------------------------------
# Sectional Time Features (Sprint 9.3)
# ---------------------------------------------------------------------------

class TestSectionalTimeFeatures:
    def setup_method(self):
        self.fb = FeatureBuilder()

    def _make_history_with_sectionals(self):
        """Create history with first_3f_secs data."""
        return pd.DataFrame({
            "horse_id": [1] * 5,
            "jockey_id": [10] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="14D").astype(str),
            "distance": [1200] * 5,
            "surface": ["turf"] * 5,
            "going": ["良"] * 5,
            "race_class": ["OP"] * 5,
            "grade": [None] * 5,
            "course_id": [5] * 5,
            "draw": [3] * 5,
            "weight_carried": [57.0] * 5,
            "horse_weight": [480] * 5,
            "odds_win": [5.0] * 5,
            "finish_pos": [1, 3, 2, 5, 1],
            "time_secs": [70.0, 71.0, 70.5, 72.0, 69.5],
            "last_3f_secs": [34.0, 34.5, 33.8, 35.0, 33.5],
            "first_3f_secs": [34.8, 35.2, 35.0, 35.5, 34.5],
            "corner_positions": ["2-1", "5-3", "3-2", "8-5", "1-1"],
            "field_size": [12] * 5,
            "margin": [0.0, 1.5, 0.5, 4.0, 0.0],
        })

    def test_first_3f_features(self):
        """Should compute avg and best first_3f from history."""
        history = self._make_history_with_sectionals()
        features = self.fb._horse_rolling_features(
            horse_id=1, race_date="2024-04-01",
            distance=1200, surface="turf", course_id=5,
            history_df=history,
        )
        assert not np.isnan(features["avg_first_3f"])
        assert not np.isnan(features["best_first_3f"])
        # Best should be <= average
        assert features["best_first_3f"] <= features["avg_first_3f"]

    def test_early_speed_derivation(self):
        """Early speed = time_secs - last_3f_secs should be computed."""
        history = self._make_history_with_sectionals()
        features = self.fb._horse_rolling_features(
            horse_id=1, race_date="2024-04-01",
            distance=1200, surface="turf", course_id=5,
            history_df=history,
        )
        assert not np.isnan(features["avg_early_speed"])
        assert not np.isnan(features["early_late_ratio_avg3"])
        # Early speed should be positive (time before last 3f)
        assert features["avg_early_speed"] > 0

    def test_sectional_features_nan_for_first_timer(self):
        """First-time runner should get NaN for sectional features."""
        empty = self.fb._empty_horse_features()
        assert np.isnan(empty["avg_first_3f"])
        assert np.isnan(empty["best_first_3f"])
        assert np.isnan(empty["avg_early_speed"])
        assert np.isnan(empty["early_late_ratio_avg3"])

    def test_first_3f_fallback_from_time_minus_last3f(self):
        """avg_first_3f/best_first_3f should be derived from time_secs - last_3f_secs
        when first_3f_secs column is absent (100% coverage fix)."""
        history = pd.DataFrame({
            "horse_id": [1] * 4,
            "jockey_id": [10] * 4,
            "date": pd.date_range("2024-01-01", periods=4, freq="14D").astype(str),
            "distance": [1200] * 4,
            "surface": ["turf"] * 4,
            "going": ["良"] * 4,
            "race_class": ["OP"] * 4,
            "grade": [None] * 4,
            "course_id": [5] * 4,
            "draw": [3] * 4,
            "weight_carried": [57.0] * 4,
            "horse_weight": [480] * 4,
            "odds_win": [5.0] * 4,
            "finish_pos": [1, 3, 2, 5],
            "time_secs": [70.0, 71.0, 70.5, 72.0],
            "last_3f_secs": [34.0, 34.5, 33.8, 35.0],
            # NO first_3f_secs column at all
            "corner_positions": ["2-1", "5-3", "3-2", "8-5"],
            "field_size": [12] * 4,
            "margin": [0.0, 1.5, 0.5, 4.0],
        })
        features = self.fb._horse_rolling_features(
            horse_id=1, race_date="2024-04-01",
            distance=1200, surface="turf", course_id=5,
            history_df=history,
        )
        # Derived first_3f = time_secs - last_3f_secs = [36.0, 36.5, 36.7, 37.0]
        assert not np.isnan(features["avg_first_3f"])
        assert not np.isnan(features["best_first_3f"])
        assert features["best_first_3f"] <= features["avg_first_3f"]
        # Verify values: 70.0-34.0=36.0, 71.0-34.5=36.5, 70.5-33.8=36.7, 72.0-35.0=37.0
        assert features["best_first_3f"] == pytest.approx(36.0, abs=0.01)
        assert features["avg_first_3f"] == pytest.approx(np.mean([36.0, 36.5, 36.7, 37.0]), abs=0.01)


# ---------------------------------------------------------------------------
# Enhanced Track Bias (Sprint 9.4) — 90-Day Rolling
# ---------------------------------------------------------------------------

class TestEnhancedTrackBias:
    def setup_method(self):
        self.fb = FeatureBuilder()

    def test_draw_bias_90d_computed(self):
        """draw_bias_90d should be computed from recent 90-day history."""
        # Create course history with draws
        history = pd.DataFrame({
            "course_id": [5] * 20,
            "draw": [3] * 10 + [7] * 10,
            "date": list(pd.date_range("2024-02-01", periods=10, freq="7D").astype(str)) +
                    list(pd.date_range("2024-02-01", periods=10, freq="7D").astype(str)),
            "finish_pos": [2, 1, 3, 1, 2, 4, 1, 3, 2, 1] +
                          [5, 6, 7, 8, 4, 9, 3, 5, 6, 7],
        })
        self.fb._course_groups = {5: history}

        row = pd.Series({
            "course_id": 5,
            "draw": 3,
            "date": "2024-06-01",
        })
        features = self.fb._track_bias_features(row, history)
        assert "draw_bias_90d" in features

    def test_draw_bias_90d_null_for_new_draw(self):
        """draw_bias_90d should be NaN for draws with < 3 observations."""
        history = pd.DataFrame({
            "course_id": [5] * 2,
            "draw": [15, 15],
            "date": ["2024-04-01", "2024-04-15"],
            "finish_pos": [1, 2],
        })
        self.fb._course_groups = {5: history}

        row = pd.Series({
            "course_id": 5,
            "draw": 15,
            "date": "2024-06-01",
        })
        features = self.fb._track_bias_features(row, history)
        assert np.isnan(features["draw_bias_90d"])

    def test_draw_bias_90d_in_empty_features(self):
        """draw_bias_90d should be in the empty features dict."""
        empty = self.fb._empty_horse_features()
        assert "draw_bias_90d" in empty
        assert np.isnan(empty["draw_bias_90d"])

