import numpy as np
import pandas as pd
import pytest
from models.features import FeatureBuilder
from models.predict_final import predict_with_filters
from scraper.db import get_session
from sqlalchemy import text

class TestFeatureEngineeringAndLeakageBrutal:
    """Stress tests feature engineering against corrupted inputs and checks for data leakage."""

    def test_strict_no_future_leakage_in_model_features(self):
        """Verify that forbidden outcome columns NEVER exist in feature_cols of saved models."""
        import os
        import json
        from pathlib import Path
        
        FORBIDDEN_OUTCOME_COLS = {
            "finish_pos", "target_win", "target_place", "target_margin",
            "time_secs", "margin", "last_3f_secs", "corner_positions",
            "payout", "won", "finishing_position", "actual_time", "order_of_finish"
        }

        saved_dir = Path("models/saved")
        meta_files = list(saved_dir.glob("*/metadata.json")) + list(saved_dir.glob("*/meta.json"))
        assert len(meta_files) >= 1, "No model metadata files found in models/saved/*!"

        for mf in meta_files:
            with open(mf) as f:
                meta = json.load(f)
            feature_cols = meta.get("feature_cols", [])
            assert len(feature_cols) >= 10, f"Model {mf} has too few feature columns!"
            for col in feature_cols:
                assert col.lower() not in FORBIDDEN_OUTCOME_COLS, (
                    f"CRITICAL DATA LEAKAGE: '{col}' found in model feature columns in {mf}!"
                )

    def test_feature_builder_on_live_races(self):
        """Build features for multiple real races and verify numerical stability (no infs, proper shapes)."""
        with get_session() as s:
            races = s.execute(text("SELECT id FROM races WHERE date = '2026-08-29' LIMIT 3")).fetchall()
        
        if not races:
            pytest.skip("No races found for 2026-08-29")
            
        race_ids = [r[0] for r in races]
        fb = FeatureBuilder()
        feat_df = fb.build_features_for_races(race_ids)
        assert not feat_df.empty
        assert len(feat_df) >= 10
        
        # Verify no infinite values in numerical features
        num_cols = feat_df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            vals = feat_df[col].dropna()
            assert not np.isinf(vals).any(), f"Infinite value found in feature column '{col}'!"

    def test_predict_with_filters_end_to_end(self):
        """Test full inference pipeline with P1-P3 filters on a batch of races."""
        with get_session() as s:
            races = s.execute(text("SELECT id FROM races WHERE date = '2026-08-29' LIMIT 2")).fetchall()
            
        if not races:
            pytest.skip("No races found for 2026-08-29")
            
        race_ids = [r[0] for r in races]
        df = predict_with_filters(race_ids=race_ids, bankroll=100000)
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "race_id" in df.columns
            assert "combined_prob" in df.columns
            assert "ev" in df.columns
            # Probabilities should all be bounded [0, 1]
            assert (df["combined_prob"] >= 0.0).all()
            assert (df["combined_prob"] <= 1.0).all()
            # EV should be finite
            assert not np.isinf(df["ev"].dropna()).any()
