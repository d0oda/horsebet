"""
Consistency test: asserts that /api/races/date/{date_str}/daily-returns matches
the exact sum of individual /api/races/{race_id}/betting-analysis calls for each strategy mode.
"""

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import text
from api.main import app
from scraper.db import get_session


@pytest.fixture
def client():
    return TestClient(app)


def test_daily_returns_reconciliation_against_race_analyses(client):
    with get_session() as s:
        # Find dates with completed races
        dates = s.execute(text("""
            SELECT r.date, COUNT(DISTINCT r.id) as num_races, COUNT(e.finish_pos) as num_finishes
            FROM races r
            JOIN entries e ON e.race_id = r.id
            WHERE e.finish_pos IS NOT NULL
            GROUP BY r.date
            HAVING COUNT(e.finish_pos) >= 10
            ORDER BY r.date DESC
            LIMIT 2
        """)).fetchall()

    if not dates:
        pytest.skip("No completed race dates in database")

    for row in dates:
        date_str = str(row[0])
        print(f"\n[INFO] Reconciling daily returns for {date_str} ({row[1]} races)...")

        # 1. Fetch daily returns API
        resp_daily = client.get(f"/api/races/date/{date_str}/daily-returns?budget_per_race=1000")
        assert resp_daily.status_code == 200
        daily_data = resp_daily.json()
        assert "strategies" in daily_data

        # 2. Fetch race IDs for this date
        with get_session() as s:
            race_ids = [
                r[0] for r in s.execute(
                    text("SELECT id FROM races WHERE date = :d ORDER BY course_id, race_number"),
                    {"d": date_str}
                ).fetchall()
            ]

        # 3. Aggregate each race across all modes
        mode_aggregates = {
            m: {"staked": 0, "payout": 0, "placed": 0, "won": 0}
            for m in ("auto", "hybrid", "pure_win", "dutching")
        }

        for race_id in race_ids:
            resp_race = client.get(f"/api/races/{race_id}/betting-analysis?budget=1000")
            if resp_race.status_code != 200:
                continue
            race_data = resp_race.json()
            all_res = race_data.get("all_strategies_results", {})

            for mode, m_res in all_res.items():
                if m_res and m_res.get("is_settled"):
                    stk = m_res.get("staked", 0)
                    pay = m_res.get("payout", 0)
                    if stk > 0:
                        mode_aggregates[mode]["staked"] += stk
                        mode_aggregates[mode]["payout"] += pay
                        mode_aggregates[mode]["placed"] += 1
                        if pay > stk:
                            mode_aggregates[mode]["won"] += 1

        # 4. Assert reconciliation against daily endpoint
        for mode, daily_mode_summary in daily_data["strategies"].items():
            agg = mode_aggregates.get(mode, {"staked": 0, "payout": 0})
            agg_profit = agg["payout"] - agg["staked"]

            print(f"  [{mode.upper()}] Daily API: Staked=¥{daily_mode_summary['staked']:,}, Payout=¥{daily_mode_summary['payout']:,}, Profit=¥{daily_mode_summary['profit']:,}")
            print(f"  [{mode.upper()}] Sum Races: Staked=¥{agg['staked']:,}, Payout=¥{agg['payout']:,}, Profit=¥{agg_profit:,}")

            assert daily_mode_summary["staked"] == agg["staked"], f"Staked mismatch in {mode} for {date_str}"
            assert daily_mode_summary["payout"] == agg["payout"], f"Payout mismatch in {mode} for {date_str}"
            assert daily_mode_summary["profit"] == agg_profit, f"Profit mismatch in {mode} for {date_str}"
