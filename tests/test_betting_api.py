"""
Integration tests for the /api/races/{race_id}/betting-analysis endpoint.
Asserts that the JSON response matches the contract specified in Section 6 of docs/BETTING_ANALYSIS_LOGIC.md.
"""

from fastapi.testclient import TestClient
import pytest
from api.main import app
from scraper.db import get_session
from sqlalchemy import text


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_race_id():
    """Finds a race in the database that has predictions."""
    with get_session() as s:
        row = s.execute(text("""
            SELECT r.id FROM races r
            JOIN predictions p ON p.race_id = r.id
            LIMIT 1
        """)).fetchone()
        if row:
            return row[0]
        # Fallback to any race if no predictions yet
        row = s.execute(text("SELECT id FROM races LIMIT 1")).fetchone()
        return row[0] if row else None


class TestBettingAnalysisAPI:
    def test_betting_analysis_success(self, client, valid_race_id):
        if valid_race_id is None:
            pytest.skip("No races in local test database")

        response = client.get(f"/api/races/{valid_race_id}/betting-analysis?budget=1000")
        assert response.status_code == 200

        data = response.json()

        # Contract assertion matching Section 6 of BETTING_ANALYSIS_LOGIC.md
        assert "race_id" in data
        assert data["race_id"] == valid_race_id
        assert "race_name" in data
        assert "budget" in data
        assert data["budget"] == 1000

        # Headline
        assert "headline" in data
        headline = data["headline"]
        assert "recommended_horse" in headline
        assert "post_position" in headline
        assert "action" in headline
        assert "summary" in headline
        assert headline["action"] == "to win"

        # Staking Plan
        assert "staking_plan" in data
        staking = data["staking_plan"]
        assert "budget" in staking
        assert staking["budget"] == 1000
        assert "tickets" in staking
        assert len(staking["tickets"]) >= 1
        assert "simple_bet_label" in staking
        assert "If you only want one uncomplicated bet" in staking["simple_bet_label"]

        # Validate ticket schema
        total_stake = 0
        for ticket in staking["tickets"]:
            assert "type" in ticket
            assert "selection" in ticket
            assert "label" in ticket
            assert "stake" in ticket
            assert "prob" in ticket
            assert "fair_odds" in ticket
            assert ticket["stake"] % 100 == 0
            total_stake += ticket["stake"]

        assert total_stake == 1000

        # Pricing Table
        assert "pricing_table" in data
        pricing_table = data["pricing_table"]
        assert len(pricing_table) >= 1

        for row in pricing_table:
            assert "post_position" in row
            assert "horse_name" in row
            assert "win_prob_pct" in row
            assert "fair_odds" in row
            assert "market_odds" in row
            assert "verdict" in row
            assert row["verdict"] in [
                "Top pick & clear value",
                "Best value",
                "Most likely, but underpriced",
                "Secondary value",
                "Longshot overlay",
                "Roughly fair",
                "Slightly short",
                "Clearly too short",
            ]

    def test_custom_budget_scaling(self, client, valid_race_id):
        if valid_race_id is None:
            pytest.skip("No races in local test database")

        response = client.get(f"/api/races/{valid_race_id}/betting-analysis?budget=5000")
        assert response.status_code == 200
        data = response.json()
        assert data["budget"] == 5000
        total_stake = sum(t["stake"] for t in data["staking_plan"]["tickets"])
        assert total_stake == 5000

    def test_nonexistent_race(self, client):
        response = client.get("/api/races/99999999/betting-analysis")
        assert response.status_code == 404

    def test_settlement_and_strategy_outcomes(self, client):
        """Tests settlement scoring and strategy comparison on a completed race."""
        with get_session() as s:
            row = s.execute(text("""
                SELECT r.id FROM races r
                JOIN entries e ON e.race_id = r.id
                WHERE e.finish_pos IS NOT NULL
                GROUP BY r.id
                HAVING COUNT(e.finish_pos) >= 5
                LIMIT 1
            """)).fetchone()

        if not row:
            pytest.skip("No completed race with results in database")

        race_id = row[0]
        response = client.get(f"/api/races/{race_id}/betting-analysis?budget=1000&mode=dutching")
        assert response.status_code == 200
        data = response.json()

        # Strategy result structure
        assert "strategy_result" in data
        s_res = data["strategy_result"]
        assert s_res["is_settled"] is True
        assert "staked" in s_res
        assert "payout" in s_res
        assert "profit" in s_res
        assert "roi_pct" in s_res
        assert "status" in s_res
        assert s_res["status"] in ("won", "lost", "push", "pass")

        # Multi-strategy comparison
        assert "all_strategies_results" in data
        all_res = data["all_strategies_results"]
        for mode in ("auto", "hybrid", "pure_win", "dutching"):
            if mode in all_res:
                assert all_res[mode]["is_settled"] is True
                assert "profit" in all_res[mode]

        # Ticket level settlement
        tickets = data["staking_plan"]["tickets"]
        for t in tickets:
            assert t["is_settled"] is True
            assert isinstance(t["won"], bool)
            assert "payout" in t
            assert "profit" in t
            assert "runner_finishes" in t
            assert len(t["runner_finishes"]) >= 1
            for rf in t["runner_finishes"]:
                assert "post_position" in rf
                assert "horse_name" in rf

        # Pricing table finish positions
        pricing_table = data["pricing_table"]
        assert any(h.get("finish_pos") is not None for h in pricing_table)


def test_settle_bet_tickets_unit():
    """Unit test for settle_bet_tickets helper across win, place, exotics."""
    from models.betting_engine import BetTicket, settle_bet_tickets

    entries_map = {
        13: {"finish_pos": 1, "odds": 4.5, "horse_name": "Bell Azzurro", "horse_name_jp": "ベルアズーロ"},
        10: {"finish_pos": 6, "odds": 16.6, "horse_name": "Cool Fidel", "horse_name_jp": "クールフィデル"},
        1:  {"finish_pos": 2, "odds": 144.1, "horse_name": "Azulite", "horse_name_jp": "アズライトアスール"},
        2:  {"finish_pos": 3, "odds": 5.2, "horse_name": "Lord Stellato", "horse_name_jp": "ロードステラート"},
    }

    # 1. Win ticket on winner (No. 13)
    t1 = BetTicket(
        ticket_type="単勝",
        ticket_type_en="win",
        selection=[13],
        selection_display="No. 13 単勝",
        prob=0.25,
        fair_odds=4.0,
        market_odds=4.5,
        stake=900,
    )

    # 2. Win ticket on 6th place (No. 10)
    t2 = BetTicket(
        ticket_type="単勝",
        ticket_type_en="win",
        selection=[10],
        selection_display="No. 10 単勝",
        prob=0.07,
        fair_odds=14.0,
        market_odds=16.6,
        stake=100,
    )

    # 3. Quinella on 13 and 1 (1st and 2nd)
    t3 = BetTicket(
        ticket_type="馬連",
        ticket_type_en="quinella",
        selection=[13, 1],
        selection_display="1–13 馬連",
        prob=0.04,
        market_odds=50.0,
        stake=200,
    )

    # 4. Wide on 13 and 10 (1st and 6th -> loss)
    t4 = BetTicket(
        ticket_type="ワイド",
        ticket_type_en="wide",
        selection=[13, 10],
        selection_display="10–13 ワイド",
        prob=0.10,
        market_odds=15.0,
        stake=100,
    )

    res = settle_bet_tickets([t1, t2, t3, t4], entries_map)

    # Assertions on tickets
    assert t1.won is True
    assert t1.payout == 4050  # 900 * 4.5
    assert t1.profit == 3150

    assert t2.won is False
    assert t2.payout == 0
    assert t2.profit == -100

    assert t3.won is True
    assert t3.payout == 10000  # 200 * 50.0
    assert t3.profit == 9800

    assert t4.won is False
    assert t4.payout == 0
    assert t4.profit == -100

    # Summary assertion
    assert res["is_settled"] is True
    assert res["staked"] == 1300
    assert res["payout"] == 14050
    assert res["profit"] == 12750
    assert res["status"] == "won"
    assert res["tickets_won"] == 2
    assert res["tickets_count"] == 4

