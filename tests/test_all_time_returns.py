import pytest
from fastapi.testclient import TestClient
from api.main import app, get_all_time_returns, _all_time_returns_cache


@pytest.fixture
def client():
    return TestClient(app)


def test_all_time_returns_success(client):
    response = client.get("/api/portfolio/all-time")
    assert response.status_code == 200
    data = response.json()

    assert "period_start" in data
    assert "period_end" in data
    assert "period_label" in data
    assert "January 2026" in data["period_label"]
    assert data["start_month"] == "2026-01"
    assert data["total_race_days"] >= 60
    assert data["total_races"] >= 2000
    assert data["total_finished_races"] >= 2000
    assert "strategies" in data
    assert "monthly_breakdown" in data
    assert "best_month" in data
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
        assert s["staked"] > 0
        assert s["profit"] == s["payout"] - s["staked"]

    # Verify monthly breakdown
    months = data["monthly_breakdown"]
    assert len(months) >= 8
    assert months[0]["month"] == "2026-01"

    # Verify sum of monthly breakdown matches all-time auto strategy
    total_staked_sum = sum(m["staked"] for m in months)
    total_payout_sum = sum(m["payout"] for m in months)
    total_profit_sum = sum(m["profit"] for m in months)
    auto_s = data["strategies"]["auto"]

    assert auto_s["staked"] == total_staked_sum
    assert auto_s["payout"] == total_payout_sum
    assert auto_s["profit"] == total_profit_sum

    # Verify best month
    best = data["best_month"]
    assert best is not None
    assert "month" in best
    assert "profit" in best
    assert best["profit"] == max(m["profit"] for m in months)


def test_all_time_returns_alias(client):
    response = client.get("/api/races/all-time-returns")
    assert response.status_code == 200
    data = response.json()
    assert data["start_month"] == "2026-01"
    assert len(data["monthly_breakdown"]) >= 8


def test_all_time_returns_date_range_filter(client):
    response = client.get("/api/portfolio/all-time?start_month=2026-02&end_month=2026-04")
    assert response.status_code == 200
    data = response.json()
    assert data["start_month"] == "2026-02"
    assert data["end_month"] == "2026-04"
    assert len(data["monthly_breakdown"]) == 3
    month_keys = [m["month"] for m in data["monthly_breakdown"]]
    assert month_keys == ["2026-02", "2026-03", "2026-04"]


def test_all_time_returns_invalid_range(client):
    response = client.get("/api/portfolio/all-time?start_month=2099-01&end_month=2099-12")
    assert response.status_code == 404
    assert "No predictions found" in response.json()["detail"]


def test_all_time_returns_direct_call():
    # Test direct python function call without FastAPI Query parameter issues
    res = get_all_time_returns(budget_per_race=1000, force_refresh=True)
    assert res is not None
    assert "strategies" in res
    assert "auto" in res["strategies"]
    assert res["strategies"]["auto"]["staked"] > 0
    assert len(res["monthly_breakdown"]) >= 8
