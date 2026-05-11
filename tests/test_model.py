"""
Unit tests for the model training pipeline.
Tests data preparation, ensemble logic, and model save/load.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.train import (
    prepare_data,
    ensemble_predict,
    evaluate_ensemble,
    MODELS_DIR,
)


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

class TestPrepareData:
    def _make_df(self, n_races=10, entries_per_race=8):
        """Create synthetic feature dataframe."""
        rows = []
        for r in range(n_races):
            for e in range(entries_per_race):
                rows.append({
                    "race_id": r + 1,
                    "entry_id": r * entries_per_race + e + 1,
                    "date": f"2024-{(r % 12) + 1:02d}-15",
                    "horse_name": f"Horse_{r}_{e}",
                    "target_win": 1 if e == 0 else 0,
                    "target_place": 1 if e < 3 else 0,
                    "finish_pos": e + 1,
                    "speed_z": np.random.randn(),
                    "form_z": np.random.randn(),
                    "odds_win": float(np.random.uniform(2.0, 50.0)),
                    "class_rank": np.random.randint(1, 10),
                    "age": np.random.randint(3, 8),
                })
        return pd.DataFrame(rows)

    def test_train_val_split(self):
        df = self._make_df(n_races=10)
        X_train, y_train, X_val, y_val, feature_cols, _ = prepare_data(df)

        # Should have some data in both splits
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_train) + len(X_val) == len(df)

    def test_target_exclusion(self):
        df = self._make_df()
        _, _, _, _, feature_cols, _ = prepare_data(df)

        assert "race_id" not in feature_cols
        assert "entry_id" not in feature_cols
        assert "target_win" not in feature_cols
        assert "target_place" not in feature_cols
        assert "finish_pos" not in feature_cols
        assert "date" not in feature_cols
        assert "horse_name" not in feature_cols

    def test_feature_cols_are_numeric(self):
        df = self._make_df()
        _, _, _, _, feature_cols, _ = prepare_data(df)

        for col in feature_cols:
            assert df[col].dtype in [np.float64, np.float32, np.int64, float, int]

    def test_nan_preserved(self):
        df = self._make_df(n_races=5)
        df.loc[0, "speed_z"] = np.nan
        df.loc[1, "speed_z"] = np.nan

        X_train, _, X_val, _, feature_cols, _ = prepare_data(df)

        # NaNs should be preserved for native tree handling
        combined = np.concatenate([X_train, X_val])
        assert np.any(np.isnan(combined))

    def test_time_based_split(self):
        df = self._make_df(n_races=10)
        # Assign sequential dates
        for i in range(10):
            mask = df["race_id"] == i + 1
            df.loc[mask, "date"] = f"2024-{i + 1:02d}-15"

        _, _, _, _, _, _ = prepare_data(df, val_date="2024-08-01")


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class TestEnsemble:
    def test_weighted_average(self):
        lgb = np.array([0.3, 0.5, 0.7])
        xgb = np.array([0.4, 0.6, 0.8])
        result = ensemble_predict(lgb, xgb, weights=(0.55, 0.45))

        expected = 0.55 * lgb + 0.45 * xgb
        np.testing.assert_array_almost_equal(result, expected)

    def test_equal_weights(self):
        lgb = np.array([0.2, 0.8])
        xgb = np.array([0.6, 0.4])
        result = ensemble_predict(lgb, xgb, weights=(0.5, 0.5))

        np.testing.assert_array_almost_equal(result, [0.4, 0.6])

    def test_full_lgb_weight(self):
        lgb = np.array([0.3, 0.7])
        xgb = np.array([0.9, 0.1])
        result = ensemble_predict(lgb, xgb, weights=(1.0, 0.0))

        np.testing.assert_array_almost_equal(result, lgb)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_evaluate_returns_metrics(self):
        y_true = np.array([1, 0, 0, 1, 0, 0, 0, 1])
        y_pred = np.array([0.8, 0.2, 0.1, 0.7, 0.3, 0.1, 0.2, 0.6])

        metrics = evaluate_ensemble(y_true, y_pred, label="Test")

        assert "logloss" in metrics
        assert "auc" in metrics
        assert "brier" in metrics
        assert metrics["logloss"] > 0
        assert 0 <= metrics["auc"] <= 1
        assert 0 <= metrics["brier"] <= 1

    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 0, 1])
        y_pred = np.array([0.99, 0.01, 0.01, 0.99])

        metrics = evaluate_ensemble(y_true, y_pred, label="Perfect")
        assert metrics["auc"] > 0.95
        assert metrics["logloss"] < 0.1

    def test_random_predictions(self):
        np.random.seed(42)
        y_true = np.random.binomial(1, 1/8, size=100)  # ~12.5% win rate
        y_pred = np.random.uniform(0, 1, size=100)

        metrics = evaluate_ensemble(y_true, y_pred, label="Random")
        # Random predictions should have poor AUC
        assert metrics["auc"] < 0.8  # Should be near 0.5 in expectation


# ---------------------------------------------------------------------------
# Sprint 2 — Exclude Features (2.1)
# ---------------------------------------------------------------------------

class TestExcludeFeatures:
    def _make_df(self, n_races=10, entries_per_race=8):
        rows = []
        for r in range(n_races):
            for e in range(entries_per_race):
                rows.append({
                    "race_id": r + 1,
                    "entry_id": r * entries_per_race + e + 1,
                    "date": f"2024-{(r % 12) + 1:02d}-15",
                    "horse_name": f"Horse_{r}_{e}",
                    "target_win": 1 if e == 0 else 0,
                    "finish_pos": e + 1,
                    "speed_z": np.random.randn(),
                    "form_z": np.random.randn(),
                    "odds_win": float(np.random.uniform(2.0, 50.0)),
                    "log_odds": float(np.random.uniform(0.5, 4.0)),
                    "popularity": np.random.randint(1, 16),
                    "odds_win_z": np.random.randn(),
                    "log_odds_z": np.random.randn(),
                    "popularity_z": np.random.randn(),
                    "class_rank": np.random.randint(1, 10),
                    "age": np.random.randint(3, 8),
                })
        return pd.DataFrame(rows)

    def test_exclude_odds_features(self):
        from models.features import ODDS_FEATURES

        df = self._make_df()
        _, _, _, _, feature_cols, _ = prepare_data(
            df, exclude_features=ODDS_FEATURES,
        )

        for feat in ODDS_FEATURES:
            assert feat not in feature_cols, f"{feat} should be excluded"

    def test_exclude_preserves_other_features(self):
        from models.features import ODDS_FEATURES

        df = self._make_df()
        _, _, _, _, feature_cols, _ = prepare_data(
            df, exclude_features=ODDS_FEATURES,
        )

        assert "speed_z" in feature_cols
        assert "form_z" in feature_cols
        assert "class_rank" in feature_cols
        assert "age" in feature_cols

    def test_exclude_none_keeps_all(self):
        df = self._make_df()
        _, _, _, _, all_cols, _ = prepare_data(df, exclude_features=None)
        _, _, _, _, exc_cols, _ = prepare_data(df, exclude_features=[])

        assert len(all_cols) == len(exc_cols)

    def test_exclude_reduces_feature_count(self):
        from models.features import ODDS_FEATURES

        df = self._make_df()
        _, _, _, _, all_cols, _ = prepare_data(df, exclude_features=None)
        _, _, _, _, odds_free_cols, _ = prepare_data(
            df, exclude_features=ODDS_FEATURES,
        )

        assert len(odds_free_cols) < len(all_cols)


# ---------------------------------------------------------------------------
# Sprint 2 — Calibration (2.3)
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_isotonic_calibrator(self):
        from models.train import calibrate_predictions

        np.random.seed(42)
        y_train = np.random.binomial(1, 0.15, size=200)
        raw_train = np.random.uniform(0, 0.5, size=200)
        y_val = np.random.binomial(1, 0.15, size=50)
        raw_val = np.random.uniform(0, 0.5, size=50)

        calibrated, calibrator = calibrate_predictions(
            y_train, raw_train, y_val, raw_val, method="isotonic"
        )

        assert calibrator is not None
        assert len(calibrated) == len(raw_val)
        assert all(0.0 <= p <= 1.0 for p in calibrated)

    def test_platt_calibrator(self):
        from models.train import calibrate_predictions

        np.random.seed(42)
        y_train = np.random.binomial(1, 0.15, size=200)
        raw_train = np.random.uniform(0.01, 0.5, size=200)
        y_val = np.random.binomial(1, 0.15, size=50)
        raw_val = np.random.uniform(0.01, 0.5, size=50)

        calibrated, calibrator = calibrate_predictions(
            y_train, raw_train, y_val, raw_val, method="platt"
        )

        assert calibrator is not None
        assert len(calibrated) == len(raw_val)
        assert all(0.0 <= p <= 1.0 for p in calibrated)

    def test_none_calibration_passthrough(self):
        from models.train import calibrate_predictions

        raw_val = np.array([0.1, 0.5, 0.9])
        calibrated, calibrator = calibrate_predictions(
            np.array([0, 1, 1]), np.array([0.2, 0.6, 0.8]),
            np.array([0, 1, 1]), raw_val, method="none"
        )

        assert calibrator is None
        np.testing.assert_array_equal(calibrated, raw_val)


# ---------------------------------------------------------------------------
# Sprint 2 — Walk-Forward CV (2.4)
# ---------------------------------------------------------------------------

class TestWalkForwardCV:
    def _make_df(self, n_races=30, entries_per_race=8):
        rows = []
        for r in range(n_races):
            for e in range(entries_per_race):
                rows.append({
                    "race_id": r + 1,
                    "entry_id": r * entries_per_race + e + 1,
                    "date": f"2024-{(r % 12) + 1:02d}-{(r // 12) + 1:02d}",
                    "horse_name": f"Horse_{r}_{e}",
                    "target_win": 1 if e == 0 else 0,
                    "speed_z": np.random.randn(),
                    "form_z": np.random.randn(),
                    "class_rank": np.random.randint(1, 10),
                })
        return pd.DataFrame(rows)

    def test_walk_forward_requires_date(self):
        from models.train import walk_forward_cv

        df = self._make_df()
        df = df.drop(columns=["date"])
        feature_cols = ["speed_z", "form_z", "class_rank"]

        with pytest.raises(ValueError, match="date"):
            walk_forward_cv(df, feature_cols)

    def test_walk_forward_returns_fold_results(self):
        from models.train import walk_forward_cv

        df = self._make_df(n_races=30)
        feature_cols = ["speed_z", "form_z", "class_rank"]

        results = walk_forward_cv(
            df, feature_cols, n_folds=3, min_train_races=5,
        )

        # Should produce some folds (depends on data size)
        assert isinstance(results, list)
        if results:
            fold = results[0]
            assert "auc" in fold
            assert "logloss" in fold
            assert "brier" in fold
            assert "n_train" in fold
            assert "n_val" in fold
            assert "val_start" in fold
            assert "val_end" in fold

    def test_walk_forward_expanding_window(self):
        """Training set should grow with each fold."""
        from models.train import walk_forward_cv

        df = self._make_df(n_races=30)
        feature_cols = ["speed_z", "form_z", "class_rank"]

        results = walk_forward_cv(
            df, feature_cols, n_folds=3, min_train_races=3,
        )

        if len(results) >= 2:
            # Each fold should have more training data than the previous
            for i in range(1, len(results)):
                assert results[i]["n_train"] >= results[i - 1]["n_train"]

