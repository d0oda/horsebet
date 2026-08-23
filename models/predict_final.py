"""
UmaEdge — Final Prediction Pipeline with P1-P3 Filters.

Configurable CLI that applies all longshot bias fixes, market reweighting,
and bet sizing logic from Priority 1-3.

Usage:
    # Predict all races for a date with default filters
    python -m models.predict_final --date 2026-02-28

    # Custom filters
    python -m models.predict_final --date 2026-02-28 \\
        --ev-threshold 0.08 --max-odds 20 --bankroll 200000

    # Single race
    python -m models.predict_final --race-id 12345 --output results.json
"""

import argparse
import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from scraper.db import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("predict_final")


def predict_with_filters(
    race_ids: list[int],
    model_version: str = "20260605_113654",
    ev_threshold: float = 0.50,
    max_odds: float = 60.0,
    min_odds: float = 2.0,
    bankroll: int = 100000,
    kelly_fraction: float = 0.25,
    flat: bool = True,
) -> pd.DataFrame:
    """
    Run predictions with all P1-P3 filters applied.

    Returns DataFrame with columns: race_id, entry_id, horse_name,
    model_prob, combined_prob, odds, market_prob, ev, kelly_frac,
    recommended_stake, is_value_bet.
    """
    from models.features import FeatureBuilder
    from models.train import load_model, ensemble_predict

    log.info(f"Loading model version: {model_version}")
    try:
        lgb_model, xgb_model, meta = load_model(version=model_version)
        feature_cols = meta["feature_cols"]
        calibrator = meta.get("calibrator")
    except FileNotFoundError:
        log.error(f"Model version '{model_version}' not found")
        return pd.DataFrame()

    all_results = []
    
    fb = FeatureBuilder()
    log.info(f"Building features for {len(race_ids)} races...")
    all_features_df = fb.build_features_for_races(race_ids)

    for race_id in race_ids:
        log.info(f"Processing race {race_id}...")

        # Build features
        features_df = all_features_df[all_features_df["race_id"] == race_id].copy() if not all_features_df.empty else pd.DataFrame()
        if features_df.empty:
            log.warning(f"No features for race {race_id}, skipping")
            continue

        # Ensure all expected feature columns exist (pre-race data may
        # be missing columns like horse_weight_z when weights aren't out).
        import xgboost as xgb_lib
        cat_cols_list = ["sire_id", "broodmare_sire_id", "going_code", "surface_code", "draw"]
        categories_map = meta.get("categories", {})
        for col in cat_cols_list:
            if col in features_df.columns:
                val = features_df[col].fillna("Unknown").astype(str)
                saved_cats = categories_map.get(col)
                if saved_cats:
                    # Enforce training category mapping so integer codes match the model
                    features_df[col] = pd.Categorical(val, categories=saved_cats)
                else:
                    features_df[col] = val.astype("category")

        for col in feature_cols:
            if col not in features_df.columns:
                log.debug(f"Adding missing column '{col}' as NaN")
                features_df[col] = np.nan

        X = features_df[feature_cols].copy()
        lgb_preds = lgb_model.predict(X)
        import re
        import xgboost.core as xgb_core
        
        # Robustly predict, stripping out unknown categories if XGBoost complains
        while True:
            try:
                dtest = xgb_lib.DMatrix(X, feature_names=feature_cols, enable_categorical=True)
                xgb_preds = xgb_model.predict(dtest)
                break
            except xgb_core.XGBoostError as e:
                err = str(e)
                m = re.search(r"for the (\d+)th \(0-based\) column: `(.+?)`", err)
                if m:
                    col_idx = int(m.group(1))
                    bad_cat = m.group(2)
                    col_name = feature_cols[col_idx]
                    log.warning(f"Removing unknown category '{bad_cat}' from {col_name}")
                    if bad_cat in X[col_name].cat.categories:
                        X[col_name] = X[col_name].cat.remove_categories([bad_cat])
                        if len(X[col_name].cat.categories) == 0:
                            cat_cols = [c for c in feature_cols if X[c].dtype.name == "category"]
                            try:
                                col_cat_idx = cat_cols.index(col_name)
                                valid_cat = lgb_model.pandas_categorical[col_cat_idx][0]
                                X[col_name] = X[col_name].cat.add_categories([valid_cat])
                            except Exception as e:
                                log.warning(f"Failed to add valid fallback category: {e}")
                    else:
                        raise e
                else:
                    raise e
        prob_cls = ensemble_predict(lgb_preds, xgb_preds)

        from models.train import scores_to_probs, cls_to_probs
        race_ids_for_probs = features_df["race_id"].values

        # Normalise binary-classifier probs per race using L1 (divide by sum).
        # Softmax would flatten calibrated probabilities; L1 correctly rescales
        # to sum-to-1 without distorting the relative magnitudes.
        prob_cls = cls_to_probs(prob_cls, race_ids_for_probs)

        # Regression blend
        lgb_reg_model = meta.get("lgb_reg_model")
        xgb_reg_model = meta.get("xgb_reg_model")
        if lgb_reg_model is not None and xgb_reg_model is not None:
            lgb_reg_preds = lgb_reg_model.predict(X)
            xgb_reg_preds = xgb_reg_model.predict(xgb_lib.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
            ensemble_reg_preds = ensemble_predict(lgb_reg_preds, xgb_reg_preds)
            # target_margin = 1/finish_pos: higher prediction → better horse.
            # Clamp to 0 to prevent any edge-case negative predictions from reg:squarederror
            # from inverting rankings (a negative prediction would rank a horse lower than
            # a correctly predicted 0.0 horse despite being near the winner).
            ensemble_reg_preds = np.maximum(ensemble_reg_preds, 0.0)
            prob_reg = scores_to_probs(ensemble_reg_preds, race_ids_for_probs)
        else:
            prob_reg = np.zeros_like(prob_cls)
            
        # Ranker blend
        lgb_rank_model = meta.get("lgb_rank_model")
        xgb_rank_model = meta.get("xgb_rank_model")
        if lgb_rank_model is not None and xgb_rank_model is not None:
            lgb_rank_preds = lgb_rank_model.predict(X)
            xgb_rank_preds = xgb_rank_model.predict(xgb_lib.DMatrix(X, feature_names=feature_cols, enable_categorical=True))
            ensemble_rank_preds = ensemble_predict(lgb_rank_preds, xgb_rank_preds)
            prob_rnk = scores_to_probs(ensemble_rank_preds, race_ids_for_probs)
        else:
            prob_rnk = np.zeros_like(prob_cls)
            
        # Goldilocks Blend (0.7 Cls, 0.2 Reg, 0.1 Rnk)
        combined = 0.7 * prob_cls + 0.2 * prob_reg + 0.1 * prob_rnk
        
        if calibrator is not None:
            from sklearn.linear_model import LogisticRegression
            if isinstance(calibrator, LogisticRegression):
                combined = calibrator.predict_proba(combined.reshape(-1, 1))[:, 1]
            else:
                combined = calibrator.predict(combined)

        fundamental = combined
        market = combined

        # Look up odds and horse names
        with get_session() as session:
            entries = session.execute(
                text("""
                    SELECT e.id, e.odds_win, h.name_jp
                    FROM entries e
                    JOIN horses h ON h.id = e.horse_id
                    WHERE e.race_id = :rid
                    ORDER BY e.post_position
                """),
                {"rid": race_id},
            ).fetchall()

        entry_map = {e[0]: {"odds": e[1], "name": e[2]} for e in entries}

        # Normalize probabilities to sum to 1.0 for the race
        comb_sum = sum(float(c) for c in combined)
        fund_sum = sum(float(f) for f in fundamental)
        mkt_sum = sum(float(m) for m in market)

        for i, (_, row) in enumerate(features_df.iterrows()):
            entry_id = row["entry_id"]
            info = entry_map.get(entry_id, {})
            odds = info.get("odds") or 0
            name = info.get("name", "?")

            combined_p = float(combined[i]) / comb_sum if comb_sum > 0 else 0
            fund_p = float(fundamental[i]) / fund_sum if fund_sum > 0 else 0
            mkt_p = float(market[i]) / mkt_sum if mkt_sum > 0 else 0

            market_prob = (1.0 / odds) if odds > 0 else 0
            ev = (combined_p * odds) - 1.0

            # Stake sizing
            if flat:
                kelly = 0
                recommended_stake = 1000
            else:
                b = odds - 1
                if b > 0 and combined_p > 0:
                    kelly = max(0, ev / b)
                    kelly *= kelly_fraction
                else:
                    kelly = 0
                recommended_stake = int(bankroll * kelly)

            # Value bet filters (P1 + P3)
            is_value = (
                ev >= ev_threshold
                and min_odds <= odds <= max_odds
                and (flat or kelly >= 0.005)  # Min Kelly fraction if not flat
            )

            all_results.append({
                "race_id": race_id,
                "entry_id": entry_id,
                "horse_name": name,
                "fundamental_prob": round(fund_p, 4),
                "market_model_prob": round(mkt_p, 4),
                "combined_prob": round(combined_p, 4),
                "odds": odds,
                "market_prob": round(market_prob, 4),
                "ev": round(ev, 4),
                "kelly_fraction": round(kelly, 4),
                "recommended_stake": recommended_stake,
                "is_value_bet": is_value,
            })

    df = pd.DataFrame(all_results)
    if not df.empty:
        # Enforce max 1 value bet per race: keep only the highest-EV bet
        value_mask = df["is_value_bet"]
        if value_mask.any():
            best_idx = (
                df[value_mask]
                .groupby("race_id")["ev"]
                .idxmax()
            )
            # Clear all value bets, then re-set only the best per race
            df["is_value_bet"] = False
            df.loc[best_idx, "is_value_bet"] = True

        df = df.sort_values(["race_id", "combined_prob"], ascending=[True, False])
    return df


def main():
    parser = argparse.ArgumentParser(
        description="UmaEdge — Filtered Prediction Pipeline (P1-P3)"
    )
    parser.add_argument("--race-id", type=int, help="Single race ID to predict")
    parser.add_argument("--date", type=str, help="Predict all races for a date (YYYY-MM-DD)")
    parser.add_argument("--version", type=str, default="20260605_113654", help="Model version")
    parser.add_argument("--ev-threshold", type=float, default=0.50, help="Min EV (default: 50%%)")
    parser.add_argument("--max-odds", type=float, default=60.0, help="Max odds (default: 60)")
    parser.add_argument("--min-odds", type=float, default=1.5, help="Min odds (default: 1.5)")
    parser.add_argument("--bankroll", type=int, default=100000, help="Bankroll in yen")
    parser.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction (default: 0.25)")
    parser.add_argument("--kelly-betting", action="store_true", help="Use Kelly betting instead of Flat (¥1000) defaults")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--value-only", action="store_true", help="Only show value bets")
    args = parser.parse_args()

    # Resolve race IDs
    if args.race_id:
        race_ids = [args.race_id]
    elif args.date:
        with get_session() as session:
            rows = session.execute(
                text("SELECT id FROM races WHERE date = :d ORDER BY race_number"),
                {"d": args.date},
            ).fetchall()
        race_ids = [r[0] for r in rows]
        if not race_ids:
            print(f"No races found for {args.date}")
            return
        log.info(f"Found {len(race_ids)} races for {args.date}")
    else:
        parser.error("Either --race-id or --date is required")
        return

    # Run predictions
    df = predict_with_filters(
        race_ids=race_ids,
        model_version=args.version,
        ev_threshold=args.ev_threshold,
        max_odds=args.max_odds,
        min_odds=args.min_odds,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        flat=not args.kelly_betting,
    )

    if df.empty:
        print("No predictions generated.")
        return

    # Filter to value bets if requested
    display_df = df[df["is_value_bet"]] if args.value_only else df

    # Display
    value_bets = df[df["is_value_bet"]]
    total_stake = value_bets["recommended_stake"].sum()

    stake_label = "Flat stake" if not args.kelly_betting else "Kelly stake"
    print(f"\n🎯 UmaEdge Predictions — {len(race_ids)} race(s)")
    print(f"   Filters: EV≥{args.ev_threshold:.0%}, odds {args.min_odds}x-{args.max_odds}x")
    print(f"   Value bets: {len(value_bets)} | Total {stake_label}: ¥{total_stake:,}")
    print("-" * 85)
    print(f"{'Horse':<16} {'Fund':>5} {'Mkt':>5} {'Comb':>5} {'Odds':>5} "
          f"{'EV':>6} {'Kelly':>6} {'Stake':>7} {'Bet':>3}")
    print("-" * 85)

    for _, row in display_df.iterrows():
        name = str(row["horse_name"])[:14]
        bet_marker = "✅" if row["is_value_bet"] else ""
        print(
            f"{name:<16} "
            f"{row['fundamental_prob']:>4.1%} "
            f"{row['market_model_prob']:>4.1%} "
            f"{row['combined_prob']:>4.1%} "
            f"{row['odds']:>5.1f} "
            f"{row['ev']:>+5.2f} "
            f"{row['kelly_fraction']:>5.2%} "
            f"¥{row['recommended_stake']:>6,} "
            f"{bet_marker:>3}"
        )

    # Save JSON if requested
    if args.output:
        output_data = {
            "filters": {
                "ev_threshold": args.ev_threshold,
                "max_odds": args.max_odds,
                "min_odds": args.min_odds,
                "bankroll": args.bankroll,
                "kelly_fraction": args.kelly,
            },
            "summary": {
                "total_races": len(race_ids),
                "total_entries": len(df),
                "value_bets": len(value_bets),
                "total_stake": int(total_stake),
            },
            "predictions": df.to_dict(orient="records"),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        log.info(f"📄 Saved to {args.output}")


if __name__ == "__main__":
    main()
