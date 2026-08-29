import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_monthly_returns_success(client):
    response = client.get("/api/races/month/2026-08/monthly-returns")
    assert response.status_code == 200
    data = response.json()
    assert data["month"] == "2026-08"
    assert "August 2026" in data["month_name"]
    assert data["total_race_days"] >= 1
    assert data["total_races"] >= 1
    assert "strategies" in data
    assert "daily_timeline" in data
    assert "available_months" in data

    # Check 4 strategy keys
    for mode in ["auto", "pure_win", "hybrid", "dutching"]:
        assert mode in data["strategies"]
        s = data["strategies"][mode]
        assert "staked" in s
        assert "payout" in s
        assert "profit" in s
        assert "roi_pct" in s
        assert "bets_placed" in s
        assert "bets_won" in s
        assert "strike_rate" in s
        assert s["profit"] == s["payout"] - s["staked"]


def test_monthly_returns_invalid_format(client):
    response = client.get("/api/races/month/invalid-month/monthly-returns")
    assert response.status_code == 400
    assert "Invalid month format" in response.json()["detail"]


def test_monthly_returns_not_found(client):
    response = client.get("/api/races/month/1999-01/monthly-returns")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "No predictions found" in detail or "No race days found" in detail


def test_portfolio_monthly_alias(client):
    response = client.get("/api/portfolio/monthly?month=2026-08")
    assert response.status_code == 200
    data = response.json()
    assert data["month"] == "2026-08"
    assert len(data["daily_timeline"]) >= 1
