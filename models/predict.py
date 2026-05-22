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

            # Align features — use NaN for missing columns (native tree handling)
            for col in feature_cols:
                if col not in features_df.columns:
                    features_df[col] = np.nan

            X = features_df[feature_cols].values
            lgb_probs = lgb_model.predict(X)
            xgb_probs = xgb_model.predict(xgb.DMatrix(X, feature_names=feature_cols))
            model_probs = ensemble_predict(lgb_probs, xgb_probs)

            # Apply calibrator if available
            calibrator = meta.get("calibrator")
            if calibrator is not None:
                model_probs = calibrator.predict(model_probs)
                log.info("Applied calibrator to predictions")
                
            # Normalize probabilities to sum to 1.0 per race (prevents IsotonicRegression inflation)
            if np.sum(model_probs) > 0:
                model_probs = model_probs / np.sum(model_probs)
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

    # 4. Look up entry data (horse_id, odds) for all entries at once
    entry_ids = features_df["entry_id"].tolist()
    with get_session() as session:
        entry_rows = session.execute(
            text("SELECT id, horse_id, odds_win FROM entries WHERE id = ANY(:ids)"),
            {"ids": entry_ids}
        ).fetchall()
    entry_map = {r[0]: (r[1], r[2] or 0) for r in entry_rows}

    # 4b. Compute overround-adjusted fair market probabilities
    all_odds = [entry_map.get(eid, (None, 0))[1] for eid in entry_ids]
    raw_implied = [(1.0 / o) if o > 0 else 0 for o in all_odds]
    overround = sum(raw_implied)
    if overround <= 0:
        overround = 1.0  # safety
    log.info(f"Market overround: {overround:.3f} ({overround*100:.1f}%)")

    # 4c. (Removed: We now read ability ratings directly from the point-in-time safe feature vector)

    # 5. Combine probabilities and apply decision labels
    result_rows = []
    for i, (_, row) in enumerate(features_df.iterrows()):
        entry_id = row["entry_id"]
        horse_id, odds = entry_map.get(entry_id, (None, 0))

        model_p = float(model_probs[i])
        fund_p = float(fundamental_probs[i]) if fundamental_probs is not None else None
        pace_p = pace_results.get(horse_id, {}).get("win_prob", model_p)
        pace_place_p = pace_results.get(horse_id, {}).get("place_prob", None)

        # Weighted combination: 60% model, 40% pace sim
        combined_win = 0.6 * model_p + 0.4 * pace_p

        # Fair market probability (overround-stripped)
        raw_prob = (1.0 / odds) if odds > 0 else 0
        fair_market_prob = raw_prob / overround if overround > 0 else 0
        edge = combined_win - fair_market_prob
        fair_odds = (1.0 / combined_win) if combined_win > 0 else 999

        # Ability rating and condition fit from feature vector
        ability_rating = row.get("horse_ability_rating", np.nan) if "horse_ability_rating" in features_df.columns else np.nan
        condition_fit = row.get("condition_fit_score", np.nan) if "condition_fit_score" in features_df.columns else np.nan
        bounce_risk = row.get("bounce_risk_flag", 0) if "bounce_risk_flag" in features_df.columns else 0

        # True Expected Value (ROI based on payout odds)
        payout_ev = (combined_win * odds) - 1.0 if odds > 0 else 0.0

        row_data = {
            "race_id": race_id,
            "entry_id": entry_id,
            "horse_id": horse_id,
            "horse_name": row.get("horse_name", ""),
            "model_win_prob": round(model_p, 4),
            "pace_win_prob": round(pace_p, 4),
            "combined_win_prob": round(combined_win, 4),
            "pace_place_prob": round(float(pace_place_p), 4) if pd.notna(pace_place_p) else None,
            "odds": odds,
            "market_prob_raw": round(raw_prob, 4),
            "fair_market_prob": round(fair_market_prob, 4),
            "overround": round(overround, 4),
            "edge": round(edge, 4),
            "fair_odds": round(fair_odds, 2),
            "ev": round(payout_ev, 4),
            "ability_rating": round(ability_rating, 1) if not pd.isna(ability_rating) else None,
            "condition_fit": round(condition_fit, 1) if not pd.isna(condition_fit) else None,
            "bounce_risk": int(bounce_risk),
            "is_value": payout_ev >= ev_threshold and min_odds <= odds <= max_odds,
        }

        if fund_p is not None:
            row_data["fundamental_win_prob"] = round(fund_p, 4)

        result_rows.append(row_data)

    result_df = pd.DataFrame(result_rows)
    result_df = result_df.sort_values("combined_win_prob", ascending=False).reset_index(drop=True)

    # 6. Apply decision labels
    result_df["decision"] = result_df.apply(
        lambda r: _decision_label(r, field_size=len(result_df)), axis=1
    )

    # 7. Store to DB
    if store_to_db:
        _store_predictions(result_df, model_version)
        _store_value_bets(result_df, model_version, ev_threshold)

    return result_df


def _decision_label(row, field_size: int) -> str:
    """Assign a decision label based on edge, condition fit, and ability rank.

    Labels (from Blueprint §7):
      Strong Value — edge >= 7%, condition_fit >= 80, high confidence
      Value        — edge >= 3%, condition_fit >= 75
      Lean         — top 3 on race-relative rank, edge >= 0
      Watch Only   — strong ability but poor conditions/bouncing
      Avoid        — everything else
    """
    edge = row.get("edge", 0) or 0
    cfit = row.get("condition_fit") or 0
    bounce = row.get("bounce_risk", 0) or 0
    ability = row.get("ability_rating") or 0
    rank = row.name + 1 if hasattr(row, "name") else field_size  # 1-based

    if edge >= 0.07 and cfit >= 80 and bounce == 0:
        return "Strong Value"
    elif edge >= 0.03 and cfit >= 75:
        return "Value"
    elif rank <= 3 and edge >= 0:
        return "Lean"
    elif ability >= 85 and (cfit < 70 or bounce == 1):
        return "Watch Only"
    else:
        return "Avoid"


def _store_predictions(df: pd.DataFrame, model_version: str):
    """Store predictions in the horsebet.predictions table."""
    with get_session() as session:
        for _, row in df.iterrows():
            session.execute(
                text("""
                    INSERT INTO predictions (
                        race_id, entry_id, model_version, win_prob, place_prob,
                        fair_odds, edge, ability_rating, condition_fit, bounce_risk, decision
                    )
                    VALUES (
                        :race_id, :entry_id, :version, :win_prob, :place_prob,
                        :fair_odds, :edge, :ability_rating, :condition_fit, :bounce_risk, :decision
                    )
                    ON CONFLICT (race_id, entry_id, model_version) DO UPDATE SET
                        win_prob = EXCLUDED.win_prob,
                        place_prob = EXCLUDED.place_prob,
                        fair_odds = EXCLUDED.fair_odds,
                        edge = EXCLUDED.edge,
                        ability_rating = EXCLUDED.ability_rating,
                        condition_fit = EXCLUDED.condition_fit,
                        bounce_risk = EXCLUDED.bounce_risk,
                        decision = EXCLUDED.decision
                """),
                {
                    "race_id": row["race_id"],
                    "entry_id": row["entry_id"],
                    "version": model_version,
                    "win_prob": row["combined_win_prob"],
                    "place_prob": row.get("pace_place_prob"),
                    "fair_odds": row.get("fair_odds"),
                    "edge": row.get("edge"),
                    "ability_rating": row.get("ability_rating"),
                    "condition_fit": row.get("condition_fit"),
                    "bounce_risk": bool(row.get("bounce_risk", 0)),
                    "decision": row.get("decision"),
                }
            )
    session.commit()
    log.info(f"💾 Stored {len(df)} predictions (version: {model_version})")


def _store_value_bets(df: pd.DataFrame, model_version: str, ev_threshold: float):
    """Store value bets in the horsebet.value_bets table."""
    value_df = df[df["is_value"]]
    if value_df.empty:
        log.info("No value bets detected for this race")
        return

    with get_session() as session:
        for _, row in value_df.iterrows():
            # Flat Betting Strategy (Top ROI Performer)
            # Replaces volatile Kelly sizing with robust ¥1,000 flat stakes
            quarter_kelly = 0.0
            recommended_stake = 1000

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
                    "market_prob": row.get("fair_market_prob", row.get("market_prob", 0)),
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
    parser.add_argument("--ev-threshold", type=float, default=0.20, help="Min EV for value bet (default: 0.20)")
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
    overround_pct = result["overround"].iloc[0] * 100 if "overround" in result.columns else 0
    print(f"\n🎯 Predictions — Race #{args.race_id}  (Overround: {overround_pct:.1f}%)")
    print("-" * 105)
    print(f"{'Horse':<14} {'Prob':>5} {'Odds':>5} {'Fair':>5} {'FairP':>5} {'Edge':>6} {'Ability':>7} {'Fit':>4} {'Decision':<12}")
    print("-" * 105)

    for _, row in result.iterrows():
        name = str(row.get("horse_name", "?"))[:12]
        ab = f"{row['ability_rating']:.0f}" if row.get('ability_rating') else "  -"
        cf = f"{row['condition_fit']:.0f}" if row.get('condition_fit') else " -"
        dec = row.get("decision", "")
        dec_icon = {"Strong Value": "🔥", "Value": "✅", "Lean": "👀", "Watch Only": "⚠️", "Avoid": ""}.get(dec, "")
        print(
            f"{name:<14} "
            f"{row['combined_win_prob']:>5.1%} "
            f"{row['odds']:>5.1f} "
            f"{row.get('fair_odds', 0):>5.1f} "
            f"{row.get('fair_market_prob', 0):>5.1%} "
            f"{row.get('edge', 0):>+5.2f} "
            f"{ab:>7} "
            f"{cf:>4} "
            f"{dec_icon} {dec}"
        )


if __name__ == "__main__":
    main()
