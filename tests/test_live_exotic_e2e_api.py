"""
End-to-End API and Multi-Threaded Concurrency Tests for Live Exotic Odds.
"""
import pytest
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from api.main import app
from scraper.db import get_session
from sqlalchemy import text


@pytest.fixture(scope="module")
def test_client():
    return TestClient(app)


def test_api_betting_analysis_structure(test_client):
    # Find a race with entries and predictions
    with get_session() as s:
        race_id = s.execute(text("""
            SELECT r.id FROM races r
            JOIN entries e ON e.race_id = r.id
            JOIN predictions p ON p.race_id = r.id
            LIMIT 1
        """)).scalar()

    if not race_id:
        pytest.skip("No races with predictions found in DB for E2E test")

    resp = test_client.get(f"/api/races/{race_id}/betting-analysis?budget=1000&strategy_mode=hybrid")
    assert resp.status_code == 200
    data = resp.json()

    assert data["race_id"] == race_id
    assert "race_name" in data
    assert "staking_plan" in data
    assert "pricing_table" in data
    assert "strategy_meta" in data

    staking = data["staking_plan"]
    assert "tickets" in staking
    assert "budget" in staking

    for ticket in staking["tickets"]:
        assert "odds_source" in ticket
        assert ticket["odds_source"] in ("live_pool", "synthetic")
        assert "market_odds" in ticket
        assert "estimated_market_odds" in ticket
        assert "fair_odds" in ticket
        assert "ev" in ticket
        assert "stake" in ticket
        assert ticket["stake"] >= 100
        assert ticket["stake"] % 100 == 0


def test_api_concurrent_requests_no_db_locks(test_client):
    """Verify that multiple concurrent requests to the betting analysis endpoint do not trigger SQLite database lock errors."""
    with get_session() as s:
        race_id = s.execute(text("""
            SELECT r.id FROM races r
            JOIN entries e ON e.race_id = r.id
            JOIN predictions p ON p.race_id = r.id
            LIMIT 1
        """)).scalar()

    if not race_id:
        pytest.skip("No races with predictions found in DB for concurrency test")

    def make_request(mode):
        return test_client.get(f"/api/races/{race_id}/betting-analysis?budget=1000&strategy_mode={mode}")

    modes = ["hybrid", "pure_win", "dutching", "auto"] * 5  # 20 concurrent requests

    with ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(executor.map(make_request, modes))

    for r in responses:
        assert r.status_code == 200, f"Failed with {r.status_code}: {r.text}"
        data = r.json()
        assert "staking_plan" in data
