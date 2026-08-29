#!/usr/bin/env python3
"""
High-performance batch reprediction for all 2026 races using active model (retrain_20260822_2143).
Processes in monthly chunks to reuse pre-indexed feature indices and minimize database round-trips.
"""
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from scraper.db import get_session
from models.predict_final import predict_with_filters
from api.main import MODEL_VERSION, _daily_returns_cache, _monthly_returns_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("repredict_2026")


def repredict_month(month_str: str, model_version: str = MODEL_VERSION, ev_threshold: float = 0.08) -> dict:
    """Predict all races for an entire month and persist to database."""
    with get_session() as s:
        race_rows = s.execute(
            text("SELECT id, race_number, course_id, date FROM races WHERE date LIKE :m ORDER BY date, course_id, race_number"),
            {"m": f"{month_str}%"},
        ).fetchall()

    if not race_rows:
        return {"races": 0, "entries": 0, "value_bets": 0}

    race_ids = [r.id for r in race_rows]
    log.info(f"Building features and evaluating {len(race_ids)} races for {month_str}...")

    t0 = time.time()
    df = predict_with_filters(
        race_ids=race_ids,
        model_version=model_version,
        ev_threshold=ev_threshold,
        max_odds=60.0,
        min_odds=2.0,
        bankroll=100000,
        kelly_fraction=0.25,
        flat=True,
    )
    eval_time = time.time() - t0

    if df.empty:
        log.warning(f"No predictions returned for {month_str}")
        return {"races": len(race_ids), "entries": 0, "value_bets": 0}

    now_iso = datetime.utcnow().isoformat()
    value_bets_count = 0
    rids_str = ",".join(str(r) for r in race_ids)

    log.info(f"Persisting {len(df)} predictions across {len(race_ids)} races to SQLite...")
    with get_session() as s:
        # Delete old predictions and value bets for these races under this model version
        s.execute(
            text(f"DELETE FROM predictions WHERE race_id IN ({rids_str}) AND model_version = :mv"),
            {"mv": model_version},
        )
        s.execute(
            text(f"DELETE FROM value_bets WHERE race_id IN ({rids_str}) AND model_version = :mv"),
            {"mv": model_version},
        )

        pred_params = []
        vb_params = []

        for _, row in df.iterrows():
            r_id = int(row["race_id"])
            e_id = int(row["entry_id"])
            m_prob = float(row["combined_prob"])
            mkt_prob = float(row["market_prob"]) if pd.notna(row.get("market_prob")) else None
            ev_val = float(row["ev"]) if pd.notna(row.get("ev")) else None
            kelly_val = float(row["kelly_fraction"]) if pd.notna(row.get("kelly_fraction")) else None
            stake_val = int(row["recommended_stake"]) if pd.notna(row.get("recommended_stake")) else 0
            is_vb = bool(row.get("is_value_bet", False))

            pred_params.append({
                "race_id": r_id,
                "entry_id": e_id,
                "win_prob": m_prob,
                "edge": ev_val,
                "mv": model_version,
                "created_at": now_iso,
            })

            if is_vb and ev_val is not None and ev_val >= ev_threshold:
                value_bets_count += 1
                vb_params.append({
                    "race_id": r_id,
                    "entry_id": e_id,
                    "model_prob": m_prob,
                    "market_prob": mkt_prob if mkt_prob is not None else 0.0,
                    "ev": ev_val,
                    "kelly": kelly_val if kelly_val is not None else 0.0,
                    "stake": stake_val,
                    "mv": model_version,
                    "created_at": now_iso,
                })

        # Batch insert chunks of 1000
        for i in range(0, len(pred_params), 1000):
            chunk = pred_params[i:i+1000]
            s.execute(
                text("""
                    INSERT INTO predictions
                        (race_id, entry_id, model_version, win_prob, edge, created_at)
                    VALUES
                        (:race_id, :entry_id, :mv, :win_prob, :edge, :created_at)
                """),
                chunk,
            )

        for i in range(0, len(vb_params), 1000):
            chunk = vb_params[i:i+1000]
            s.execute(
                text("""
                    INSERT INTO value_bets
                        (race_id, entry_id, bet_type, model_prob, market_prob,
                         ev, kelly_fraction, recommended_stake, model_version, created_at)
                    VALUES
                        (:race_id, :entry_id, 'win', :model_prob, :market_prob,
                         :ev, :kelly, :stake, :mv, :created_at)
                """),
                chunk,
            )

        s.commit()

    log.info(f"✅ {month_str}: {len(race_ids)} races ({len(df)} entries) → {value_bets_count} value bets (eval={eval_time:.1f}s)")
    return {"races": len(race_ids), "entries": len(df), "value_bets": value_bets_count}


def main():
    months = [f"2026-{m:02d}" for m in range(1, 9)]
    log.info(f"━━━ Starting Full 2026 Batch Reprediction across {len(months)} Months (Model: {MODEL_VERSION}) ━━━")
    t0_all = time.time()

    total_races = 0
    total_entries = 0
    total_vb = 0

    for i, m_str in enumerate(months, 1):
        log.info(f"━━━━━━━━ [{i}/{len(months)}] Month {m_str} ━━━━━━━━")
        try:
            res = repredict_month(m_str, model_version=MODEL_VERSION)
            total_races += res["races"]
            total_entries += res["entries"]
            total_vb += res["value_bets"]
        except Exception as e:
            log.error(f"❌ Failed to process month {m_str}: {e}")
            import traceback
            traceback.print_exc()

    # Clear caches
    _daily_returns_cache.clear()
    _monthly_returns_cache.clear()

    total_time = time.time() - t0_all
    log.info(f"━━━ Complete! Repredicted {total_races} races ({total_entries} entries, {total_vb} value bets) in {total_time/60:.1f}m ━━━")


if __name__ == "__main__":
    main()
