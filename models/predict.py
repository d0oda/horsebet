"""
UmaEdge — End-to-End Prediction Pipeline.

Loads a trained model, builds features for a race, runs pace simulation,
combines probabilities, and stores predictions + value bets in the database.

Usage:
    # Predict for a specific race
    python -m models.predict --race-id 1

    # Predict with a specific model version
    python -m models.predict --race-id 1 --version 20250101_120000

    # Predict without storing to DB
    python -m models.predict --race-id 1 --no-store
"""

import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from scraper.db import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("predict")


def predict_and_store(
    race_id: int,
    model_version: str = "latest",
    store_to_db: bool = True,
    ev_threshold: float = 0.30,
    use_hybrid: bool = False,
    max_odds: float = 30.0,
    min_odds: float = 2.0,
) -> pd.DataFrame:
    """
    Full prediction pipeline for a race:
    1. Build features from DB
    2. Run model inference (LightGBM + XGBoost ensemble)
    3. Run pace simulation
    4. Combine model + pace sim probabilities
    5. Detect value bets (model_prob > implied_prob + threshold)
    6. Store predictions and value bets to DB

    Args:
        race_id: Database race ID.
        model_version: Model version to use.
        store_to_db: Whether to persist predictions.
        ev_threshold: Minimum EV to flag as value bet.
        use_hybrid: If True, use HybridEnsemble for divergence-based value.

    Returns:
        DataFrame with predictions per entry
    """
    from models.features import FeatureBuilder
    from models.train import load_model, ensemble_predict
    from models.simulate_race import simulate_race_by_id

    # 1. Build features
    log.info(f"Building features for race {race_id}...")
    fb = FeatureBuilder()
    features_df = fb.build_features_for_race(race_id)
    if features_df.empty:
        log.error(f"No features could be built for race {race_id}")
        return pd.DataFrame()

    # 2. Model inference
    log.info("Running model inference...")
    fundamental_probs = None

    if use_hybrid:
        # Hybrid ensemble: fundamental + market models
        try:
            from models.ensemble import HybridEnsemble, DivergenceDetector
            hybrid = HybridEnsemble.load(version=model_version if model_version != "latest" else "hybrid")
            preds = hybrid.predict(features_df)
            model_probs = preds["combined"]
            fundamental_probs = preds["fundamental"]
            log.info("Using hybrid ensemble (fundamental + market)")
        except (FileNotFoundError, Exception) as e:
            log.warning(f"Hybrid model not available ({e}), falling back to standard model")
            use_hybrid = False

    if not use_hybrid:
        try:
            import xgboost as xgb
            lgb_model, xgb_model, meta = load_model(model_version)
            feature_cols = meta["feature_cols"]

            # Align features — use 0 for missing columns
            for col in feature_cols:
                if col not in features_df.columns:
                    features_df[col] = 0

            X = features_df[feature_cols].fillna(0).values
            lgb_probs = lgb_model.predict(X)
            xgb_probs = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_cols))
            model_probs = ensemble_predict(lgb_probs, xgb_probs)

            # Apply calibrator if available
            calibrator = meta.get("calibrator")
            if calibrator is not None:
                model_probs = calibrator.predict(model_probs)
                log.info("Applied calibrator to predictions")
        except FileNotFoundError:
            log.warning("No trained model found — using uniform probabilities")
            n = len(features_df)
            model_probs = np.ones(n) / n

    # 3. Pace simulation
    log.info("Running pace simulation...")
    try:
        pace_results = simulate_race_by_id(race_id)
    except Exception as e:
        log.warning(f"Pace simulation failed: {e}")
        pace_results = {}

    # 4. Combine probabilities
    result_rows = []
    for i, (_, row) in enumerate(features_df.iterrows()):
        entry_id = row["entry_id"]
        horse_id = None

        # Look up horse_id from entry
        with get_session() as session:
            entry_data = session.execute(
                text("SELECT horse_id, odds_win FROM entries WHERE id = :eid"),
                {"eid": entry_id}
            ).fetchone()

        if entry_data:
            horse_id = entry_data[0]
            odds = entry_data[1] or 0
        else:
            odds = 0

        model_p = float(model_probs[i])
        fund_p = float(fundamental_probs[i]) if fundamental_probs is not None else None
        pace_p = pace_results.get(horse_id, {}).get("win_prob", model_p)
        pace_place_p = pace_results.get(horse_id, {}).get("place_prob", None)

        # Weighted combination: 60% model, 40% pace sim
        combined_win = 0.6 * model_p + 0.4 * pace_p

        # Market implied probability
        market_prob = (1.0 / odds) if odds > 0 else 0
        ev = combined_win - market_prob

        row_data = {
            "race_id": race_id,
            "entry_id": entry_id,
            "horse_name": row.get("horse_name", ""),
            "model_win_prob": round(model_p, 4),
            "pace_win_prob": round(pace_p, 4),
            "combined_win_prob": round(combined_win, 4),
            "pace_place_prob": round(pace_place_p, 4) if pace_place_p else None,
            "odds": odds,
            "market_prob": round(market_prob, 4),
            "ev": round(ev, 4),
            "is_value": ev >= ev_threshold and min_odds <= odds <= max_odds,
        }

        if fund_p is not None:
            row_data["fundamental_win_prob"] = round(fund_p, 4)

        result_rows.append(row_data)

    result_df = pd.DataFrame(result_rows)
    result_df = result_df.sort_values("combined_win_prob", ascending=False)

    # 5. Store to DB
    if store_to_db:
        _store_predictions(result_df, model_version)
        _store_value_bets(result_df, model_version, ev_threshold)

    return result_df


def _store_predictions(df: pd.DataFrame, model_version: str):
    """Store predictions in the horsebet.predictions table."""
    with get_session() as session:
        for _, row in df.iterrows():
            session.execute(
                text("""
                    INSERT INTO predictions (race_id, entry_id, model_version, win_prob, place_prob)
                    VALUES (:race_id, :entry_id, :version, :win_prob, :place_prob)
                    ON CONFLICT (race_id, entry_id, model_version) DO UPDATE SET
                        win_prob = EXCLUDED.win_prob,
                        place_prob = EXCLUDED.place_prob
                """),
                {
                    "race_id": row["race_id"],
                    "entry_id": row["entry_id"],
                    "version": model_version,
                    "win_prob": row["combined_win_prob"],
                    "place_prob": row.get("pace_place_prob"),
                }
            )
    log.info(f"💾 Stored {len(df)} predictions (version: {model_version})")


def _store_value_bets(df: pd.DataFrame, model_version: str, ev_threshold: float):
    """Store value bets in the horsebet.value_bets table."""
    value_df = df[df["is_value"]]
    if value_df.empty:
        log.info("No value bets detected for this race")
        return

    with get_session() as session:
        for _, row in value_df.iterrows():
            # Kelly sizing
            b = row["odds"] - 1
            p = row["combined_win_prob"]
            q = 1 - p
            kelly = max(0, (b * p - q) / b) if b > 0 else 0
            quarter_kelly = kelly * 0.25
            recommended_stake = int(100000 * quarter_kelly)  # assume 100k bankroll

            session.execute(
                text("""
                    INSERT INTO value_bets (
                        race_id, entry_id, bet_type, model_prob,
                        market_prob, ev, kelly_fraction, recommended_stake,
                        model_version
                    ) VALUES (
                        :race_id, :entry_id, 'win', :model_prob,
                        :market_prob, :ev, :kelly, :stake,
                        :version
                    )
                """),
                {
                    "race_id": row["race_id"],
                    "entry_id": row["entry_id"],
                    "model_prob": row["combined_win_prob"],
                    "market_prob": row["market_prob"],
                    "ev": row["ev"],
                    "kelly": quarter_kelly,
                    "stake": recommended_stake,
                    "version": model_version,
                }
            )

    log.info(f"💰 Stored {len(value_df)} value bets")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Prediction Pipeline")
    parser.add_argument("--race-id", type=int, required=True, help="Database race ID")
    parser.add_argument("--version", type=str, default="latest", help="Model version")
    parser.add_argument("--no-store", action="store_true", help="Don't store to DB")
    parser.add_argument("--ev-threshold", type=float, default=0.05, help="Min EV for value bet")
    parser.add_argument("--max-odds", type=float, default=30.0, help="Max odds to bet (default: 30)")
    parser.add_argument("--min-odds", type=float, default=1.5, help="Min odds to bet (default: 1.5)")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid ensemble")
    args = parser.parse_args()

    result = predict_and_store(
        race_id=args.race_id,
        model_version=args.version,
        store_to_db=not args.no_store,
        ev_threshold=args.ev_threshold,
        use_hybrid=args.hybrid,
        max_odds=args.max_odds,
        min_odds=args.min_odds,
    )

    if result.empty:
        print("No predictions generated.")
        return

    # Display
    print(f"\n🎯 Predictions — Race #{args.race_id}")
    print("-" * 75)
    print(f"{'Horse':<16} {'Model':>6} {'Pace':>6} {'Combined':>8} {'Odds':>5} {'MktP':>5} {'EV':>6} {'Value':>5}")
    print("-" * 75)

    for _, row in result.iterrows():
        name = str(row.get("horse_name", "?"))[:14]
        is_val = "✅" if row["is_value"] else ""
        print(
            f"{name:<16} "
            f"{row['model_win_prob']:>5.1%} "
            f"{row['pace_win_prob']:>5.1%} "
            f"{row['combined_win_prob']:>7.1%} "
            f"{row['odds']:>5.1f} "
            f"{row['market_prob']:>5.1%} "
            f"{row['ev']:>+5.2f} "
            f"{is_val:>5}"
        )


if __name__ == "__main__":
    main()
