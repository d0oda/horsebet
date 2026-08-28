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
