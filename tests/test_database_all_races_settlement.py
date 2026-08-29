"""
Massive verification test running betting engine settlement across ALL completed races in the database.
"""

import pytest
from sqlalchemy import text
from scraper.db import get_session
from models.betting_engine import analyze_race_betting


def test_all_completed_races_in_database():
    with get_session() as s:
        # Find all race IDs that have at least 1 finish position
        races = s.execute(text("""
            SELECT r.id, r.date, r.race_number, c.name, COUNT(e.finish_pos) as num_finishes
            FROM races r
            JOIN entries e ON e.race_id = r.id
            LEFT JOIN courses c ON c.id = r.course_id
            WHERE e.finish_pos IS NOT NULL
            GROUP BY r.id, r.date, r.race_number, c.name
            HAVING COUNT(e.finish_pos) >= 3
            ORDER BY r.date DESC, r.id DESC
            LIMIT 50
        """)).fetchall()

    if not races:
        pytest.skip("No completed races found in local test DB")

    print(f"\n[INFO] Running brutal settlement verification on {len(races)} historical races...")

    modes = ["auto", "hybrid", "pure_win", "dutching"]
    budgets = [500, 1000, 2500]

    verified_count = 0
    ticket_count = 0
    winning_ticket_count = 0

    with get_session() as s:
        for r in races:
            race_id = r[0]
            for budget in budgets:
                for mode in modes:
                    data = analyze_race_betting(
                        race_id=race_id,
                        budget=budget,
                        strategy_mode=mode,
                        session=s,
                    )

                    # 1. Structure validation
                    assert "strategy_result" in data
                    assert "all_strategies_results" in data
                    assert "staking_plan" in data
                    assert "pricing_table" in data

                    s_res = data["strategy_result"]
                    assert s_res["is_settled"] is True
                    assert s_res["payout"] >= 0
                    assert s_res["staked"] >= 0
                    assert s_res["profit"] == s_res["payout"] - s_res["staked"]
                    assert s_res["status"] in ("won", "lost", "push", "pass")

                    tickets = data["staking_plan"]["tickets"]
                    ticket_sum_stake = sum(t["stake"] for t in tickets)
                    ticket_sum_payout = sum(t["payout"] for t in tickets)
                    ticket_sum_profit = sum(t["profit"] for t in tickets)
                    ticket_sum_won = sum(1 for t in tickets if t["won"])

                    # Staking budget invariant
                    assert ticket_sum_stake == s_res["staked"]
                    assert ticket_sum_stake <= budget
                    assert ticket_sum_payout == s_res["payout"]
                    assert ticket_sum_profit == s_res["profit"]
                    assert ticket_sum_won == s_res["tickets_won"]

                    for t in tickets:
                        ticket_count += 1
                        if t["won"]:
                            winning_ticket_count += 1
                        assert t["is_settled"] is True
                        assert isinstance(t["won"], bool)
                        assert t["payout"] >= 0
                        assert t["profit"] == t["payout"] - t["stake"]
                        assert len(t["runner_finishes"]) >= 1

                    # Finish position consistency on pricing table
                    for h in data["pricing_table"]:
                        assert "finish_pos" in h

            verified_count += 1

    print(f"\n[INFO] Verified {verified_count} races ({verified_count * len(budgets) * len(modes)} evaluations).")
    print(f"[INFO] Verified {ticket_count} total tickets ({winning_ticket_count} winning tickets).")
