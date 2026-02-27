"""
UmaEdge — Hybrid Ensemble (Odds-Free + Odds-Aware).

Sprint 2.2: Combines a "fundamental" model (trained without any odds
features) with the standard "market-aware" model. The edge comes from
cases where the fundamental model disagrees with the market — this
suggests the horse has value that the crowd hasn't fully priced in.

Usage:
    from models.ensemble import HybridEnsemble

    hybrid = HybridEnsemble()
    hybrid.train(df, val_date="2025-01-01")
    result = hybrid.predict(X, feature_cols)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from models.features import ODDS_FEATURES
from models.train import (
    prepare_data,
    train_lightgbm,
    train_xgboost,
    ensemble_predict,
    evaluate_ensemble,
    calibrate_predictions,
    save_model,
    load_model,
    MODELS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ensemble")


# ---------------------------------------------------------------------------
# Divergence Detector
# ---------------------------------------------------------------------------

@dataclass
class AlphaSignal:
    """A detected divergence between fundamental and market models."""
    entry_id: int
    fundamental_prob: float
    market_prob: float
    odds_aware_prob: float
    divergence: float  # fundamental - market (positive = value)
    alpha_score: float  # combined confidence metric


class DivergenceDetector:
    """
    Detects entries where the fundamental model (odds-free) significantly
    disagrees with the market (implied odds probability).

    If the fundamental model predicts a horse has a higher chance of winning
    than the market implies, this is a potential alpha opportunity.
    """

    def __init__(self, min_divergence: float = 0.05):
        """
        Args:
            min_divergence: Minimum divergence (fundamental_prob - market_prob)
                to flag as an alpha signal.
        """
        self.min_divergence = min_divergence

    def detect(
        self,
        entry_ids: np.ndarray,
        fundamental_probs: np.ndarray,
        odds_aware_probs: np.ndarray,
        market_odds: np.ndarray,
    ) -> list[AlphaSignal]:
        """
        Detect alpha signals where fundamental model disagrees with market.

        Args:
            entry_ids: Array of entry IDs.
            fundamental_probs: Predictions from the odds-free model.
            odds_aware_probs: Predictions from the odds-aware model.
            market_odds: Win odds from the market.

        Returns:
            List of AlphaSignal objects, sorted by alpha_score descending.
        """
        signals = []

        for i in range(len(entry_ids)):
            odds = market_odds[i]
            if not odds or odds <= 0:
                continue

            market_prob = 1.0 / odds
            fundamental_p = fundamental_probs[i]
            odds_aware_p = odds_aware_probs[i]

            # Divergence: fundamental model says horse is stronger than market
            divergence = fundamental_p - market_prob

            if divergence < self.min_divergence:
                continue

            # Alpha score: combines divergence magnitude with model confidence
            # Higher when both models see some probability AND there's divergence
            model_agreement = 1.0 - abs(fundamental_p - odds_aware_p)
            alpha_score = divergence * (0.7 + 0.3 * model_agreement)

            signals.append(AlphaSignal(
                entry_id=int(entry_ids[i]),
                fundamental_prob=round(float(fundamental_p), 4),
                market_prob=round(float(market_prob), 4),
                odds_aware_prob=round(float(odds_aware_p), 4),
                divergence=round(float(divergence), 4),
                alpha_score=round(float(alpha_score), 4),
            ))

        # Sort by alpha score, highest first
        signals.sort(key=lambda s: s.alpha_score, reverse=True)
        return signals


# ---------------------------------------------------------------------------
# Hybrid Ensemble
# ---------------------------------------------------------------------------

class HybridEnsemble:
    """
    Combines an odds-free (fundamental) model with an odds-aware (market)
    model. The odds-free model discovers intrinsic horse quality from speed,
    form, jockey, and pace data. The odds-aware model captures market
    consensus. EV comes from cases where the fundamental model sees value
    that the market has not priced in.
    """

    def __init__(
        self,
        fundamental_weight: float = 0.6,
        market_weight: float = 0.4,
        calibration_method: str = "isotonic",
    ):
        """
        Args:
            fundamental_weight: Weight for odds-free model in combined prediction.
            market_weight: Weight for odds-aware model in combined prediction.
            calibration_method: Calibration to apply ('none', 'platt', 'isotonic').
        """
        self.fundamental_weight = fundamental_weight
        self.market_weight = market_weight
        self.calibration_method = calibration_method

        # Models (set after training)
        self.fund_lgb = None
        self.fund_xgb = None
        self.fund_feature_cols = None
        self.fund_calibrator = None

        self.mkt_lgb = None
        self.mkt_xgb = None
        self.mkt_feature_cols = None
        self.mkt_calibrator = None

    def train(
        self,
        df: pd.DataFrame,
        val_date: Optional[str] = None,
        target: str = "target_win",
    ) -> dict:
        """
        Train both the fundamental (odds-free) and market (odds-aware) models.

        Args:
            df: Full feature dataframe from FeatureBuilder.
            val_date: Validation split date.
            target: Target column.

        Returns:
            Dict with metrics for both models.
        """
        results = {}

        # --- Fundamental model (odds-free) ---
        log.info("\n" + "=" * 60)
        log.info("  Training FUNDAMENTAL model (odds-free)")
        log.info("=" * 60)

        X_train_f, y_train_f, X_val_f, y_val_f, fund_cols, _ = prepare_data(
            df.copy(), target=target, val_date=val_date,
            exclude_features=ODDS_FEATURES,
        )

        lgb_f, lgb_preds_f = train_lightgbm(X_train_f, y_train_f, X_val_f, y_val_f, fund_cols)
        xgb_f, xgb_preds_f = train_xgboost(X_train_f, y_train_f, X_val_f, y_val_f, fund_cols)
        ens_preds_f = ensemble_predict(lgb_preds_f, xgb_preds_f)

        # Calibrate fundamental model
        if self.calibration_method != "none":
            lgb_train_f = lgb_f.predict(X_train_f)
            import xgboost as xgb_lib
            xgb_train_f = xgb_f.predict(
                xgb_lib.DMatrix(X_train_f, feature_names=fund_cols)
            )
            ens_train_f = ensemble_predict(lgb_train_f, xgb_train_f)
            ens_preds_f, self.fund_calibrator = calibrate_predictions(
                y_train_f, ens_train_f, y_val_f, ens_preds_f,
                method=self.calibration_method,
            )

        fund_metrics = evaluate_ensemble(y_val_f, ens_preds_f, label="Fundamental (odds-free)")
        results["fundamental"] = fund_metrics

        self.fund_lgb = lgb_f
        self.fund_xgb = xgb_f
        self.fund_feature_cols = fund_cols

        # --- Market model (odds-aware) ---
        log.info("\n" + "=" * 60)
        log.info("  Training MARKET model (odds-aware)")
        log.info("=" * 60)

        X_train_m, y_train_m, X_val_m, y_val_m, mkt_cols, race_ids_val = prepare_data(
            df.copy(), target=target, val_date=val_date,
        )

        lgb_m, lgb_preds_m = train_lightgbm(X_train_m, y_train_m, X_val_m, y_val_m, mkt_cols)
        xgb_m, xgb_preds_m = train_xgboost(X_train_m, y_train_m, X_val_m, y_val_m, mkt_cols)
        ens_preds_m = ensemble_predict(lgb_preds_m, xgb_preds_m)

        # Calibrate market model
        if self.calibration_method != "none":
            lgb_train_m = lgb_m.predict(X_train_m)
            import xgboost as xgb_lib
            xgb_train_m = xgb_m.predict(
                xgb_lib.DMatrix(X_train_m, feature_names=mkt_cols)
            )
            ens_train_m = ensemble_predict(lgb_train_m, xgb_train_m)
            ens_preds_m, self.mkt_calibrator = calibrate_predictions(
                y_train_m, ens_train_m, y_val_m, ens_preds_m,
                method=self.calibration_method,
            )

        mkt_metrics = evaluate_ensemble(y_val_m, ens_preds_m, label="Market (odds-aware)")
        results["market"] = mkt_metrics

        self.mkt_lgb = lgb_m
        self.mkt_xgb = xgb_m
        self.mkt_feature_cols = mkt_cols

        # --- Divergence analysis on validation set ---
        log.info("\n" + "=" * 60)
        log.info("  Divergence Analysis (fundamental vs market)")
        log.info("=" * 60)

        divergence = ens_preds_f - ens_preds_m
        log.info(f"Mean divergence:    {divergence.mean():+.4f}")
        log.info(f"Std divergence:     {divergence.std():.4f}")
        log.info(f"Max divergence:     {divergence.max():+.4f}")
        log.info(f"Min divergence:     {divergence.min():+.4f}")
        log.info(f"Entries w/ div>0.05: {(divergence > 0.05).sum()}")
        log.info(f"Entries w/ div>0.10: {(divergence > 0.10).sum()}")

        results["divergence_stats"] = {
            "mean": float(divergence.mean()),
            "std": float(divergence.std()),
            "max": float(divergence.max()),
            "min": float(divergence.min()),
            "n_above_5pct": int((divergence > 0.05).sum()),
            "n_above_10pct": int((divergence > 0.10).sum()),
        }

        return results

    def predict(
        self,
        features_df: pd.DataFrame,
    ) -> dict:
        """
        Generate predictions from both models for a set of entries.

        Args:
            features_df: Feature dataframe for entries to predict.

        Returns:
            Dict with 'fundamental', 'market', 'combined' prediction arrays.
        """
        import xgboost as xgb_lib

        # Fundamental predictions
        X_fund = features_df[self.fund_feature_cols].fillna(0).values
        lgb_f = self.fund_lgb.predict(X_fund)
        xgb_f = self.fund_xgb.predict(
            xgb_lib.DMatrix(X_fund, feature_names=self.fund_feature_cols)
        )
        fund_preds = ensemble_predict(lgb_f, xgb_f)

        if self.fund_calibrator is not None:
            from sklearn.linear_model import LogisticRegression
            if isinstance(self.fund_calibrator, LogisticRegression):
                fund_preds = self.fund_calibrator.predict_proba(fund_preds.reshape(-1, 1))[:, 1]
            else:
                fund_preds = self.fund_calibrator.predict(fund_preds)

        # Market predictions
        X_mkt = features_df[self.mkt_feature_cols].fillna(0).values
        lgb_m = self.mkt_lgb.predict(X_mkt)
        xgb_m = self.mkt_xgb.predict(
            xgb_lib.DMatrix(X_mkt, feature_names=self.mkt_feature_cols)
        )
        mkt_preds = ensemble_predict(lgb_m, xgb_m)

        if self.mkt_calibrator is not None:
            from sklearn.linear_model import LogisticRegression
            if isinstance(self.mkt_calibrator, LogisticRegression):
                mkt_preds = self.mkt_calibrator.predict_proba(mkt_preds.reshape(-1, 1))[:, 1]
            else:
                mkt_preds = self.mkt_calibrator.predict(mkt_preds)

        # Combined prediction: weighted blend
        combined = (
            self.fundamental_weight * fund_preds
            + self.market_weight * mkt_preds
        )

        return {
            "fundamental": fund_preds,
            "market": mkt_preds,
            "combined": combined,
        }

    def save(self, version: str = "hybrid") -> str:
        """Save both sub-models under one version directory."""
        from pathlib import Path
        import json
        import pickle

        model_dir = MODELS_DIR / version
        model_dir.mkdir(exist_ok=True)

        # Save fundamental
        fund_dir = model_dir / "fundamental"
        fund_dir.mkdir(exist_ok=True)
        self.fund_lgb.save_model(str(fund_dir / "lgb_model.txt"))
        self.fund_xgb.save_model(str(fund_dir / "xgb_model.json"))
        if self.fund_calibrator:
            with open(fund_dir / "calibrator.pkl", "wb") as f:
                pickle.dump(self.fund_calibrator, f)

        # Save market
        mkt_dir = model_dir / "market"
        mkt_dir.mkdir(exist_ok=True)
        self.mkt_lgb.save_model(str(mkt_dir / "lgb_model.txt"))
        self.mkt_xgb.save_model(str(mkt_dir / "xgb_model.json"))
        if self.mkt_calibrator:
            with open(mkt_dir / "calibrator.pkl", "wb") as f:
                pickle.dump(self.mkt_calibrator, f)

        # Save metadata
        meta = {
            "version": version,
            "type": "hybrid",
            "fundamental_feature_cols": self.fund_feature_cols,
            "market_feature_cols": self.mkt_feature_cols,
            "fundamental_weight": self.fundamental_weight,
            "market_weight": self.market_weight,
            "calibration_method": self.calibration_method,
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        log.info(f"💾 Hybrid ensemble saved to {model_dir}")
        return version

    @classmethod
    def load(cls, version: str = "hybrid") -> "HybridEnsemble":
        """Load a saved hybrid ensemble."""
        import json
        import pickle
        import lightgbm as lgb_lib
        import xgboost as xgb_lib

        model_dir = MODELS_DIR / version

        with open(model_dir / "metadata.json") as f:
            meta = json.load(f)

        hybrid = cls(
            fundamental_weight=meta["fundamental_weight"],
            market_weight=meta["market_weight"],
            calibration_method=meta.get("calibration_method", "none"),
        )

        # Load fundamental
        fund_dir = model_dir / "fundamental"
        hybrid.fund_lgb = lgb_lib.Booster(model_file=str(fund_dir / "lgb_model.txt"))
        hybrid.fund_xgb = xgb_lib.Booster()
        hybrid.fund_xgb.load_model(str(fund_dir / "xgb_model.json"))
        hybrid.fund_feature_cols = meta["fundamental_feature_cols"]

        cal_path = fund_dir / "calibrator.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                hybrid.fund_calibrator = pickle.load(f)

        # Load market
        mkt_dir = model_dir / "market"
        hybrid.mkt_lgb = lgb_lib.Booster(model_file=str(mkt_dir / "lgb_model.txt"))
        hybrid.mkt_xgb = xgb_lib.Booster()
        hybrid.mkt_xgb.load_model(str(mkt_dir / "xgb_model.json"))
        hybrid.mkt_feature_cols = meta["market_feature_cols"]

        cal_path = mkt_dir / "calibrator.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                hybrid.mkt_calibrator = pickle.load(f)

        log.info(f"📦 Loaded hybrid ensemble version: {version}")
        return hybrid
