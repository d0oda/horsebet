import concurrent.futures
import pytest
from fastapi.testclient import TestClient
from api.main import app, _all_time_returns_cache


@pytest.fixture
def client():
    return TestClient(app)


def test_brutal_concurrency_stress(client):
    """Fires 20 concurrent requests with varying params to test thread-safety and cache stability."""
    # Warm up first
    client.get("/api/portfolio/all-time")

    endpoints = [
        "/api/portfolio/all-time",
        "/api/portfolio/all-time?budget_per_race=500",
        "/api/portfolio/all-time?budget_per_race=2000",
        "/api/portfolio/all-time?start_month=2026-02&end_month=2026-06",
        "/api/portfolio/all-time?start_month=2026-03&end_month=2026-03",
        "/api/races/all-time-returns",
        "/api/races/all-time/returns",
        "/api/portfolio/monthly?month=2026-03",
        "/api/portfolio/monthly?month=2026-08",
    ] * 2

    def fetch_url(url):
        res = client.get(url)
        return res.status_code, url

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_url, u) for u in endpoints]
        results = [f.result() for f in futures]

    for status_code, url in results:
        assert status_code == 200, f"Expected 200 for {url}, got {status_code}"


def test_brutal_parameter_boundaries(client):
    """Test extreme boundaries, invalid budgets, and malformed inputs."""
    # Negative budget
    res = client.get("/api/portfolio/all-time?budget_per_race=-500")
    assert res.status_code == 422

    # Excessively large budget
    res = client.get("/api/portfolio/all-time?budget_per_race=999999999999")
    assert res.status_code == 422

    # Minimum budget allowed (100 yen)
    res = client.get("/api/portfolio/all-time?budget_per_race=100")
    assert res.status_code == 200
    data = res.json()
    assert data["budget_per_race"] == 100

    # Inverted range (start > end)
    res = client.get("/api/portfolio/all-time?start_month=2026-08&end_month=2026-01")
    assert res.status_code == 404
    assert "No predictions found in range" in res.json()["detail"]

    # Non-existent month
    res = client.get("/api/portfolio/all-time?start_month=1990-01&end_month=1990-12")
    assert res.status_code == 404


def test_brutal_mathematical_precision(client):
    """Validate mathematical integrity of every single strategy and month."""
    res = client.get("/api/portfolio/all-time?budget_per_race=1000")
    assert res.status_code == 200
    data = res.json()

    for strat_key, strat in data["strategies"].items():
        staked = strat["staked"]
        payout = strat["payout"]
        profit = strat["profit"]
        bets_placed = strat["bets_placed"]
        bets_won = strat["bets_won"]
        strike_rate = strat["strike_rate"]
        roi_pct = strat["roi_pct"]

        # Math checks
        assert profit == payout - staked, f"{strat_key}: profit {profit} != payout {payout} - staked {staked}"
        assert bets_won <= bets_placed, f"{strat_key}: bets_won {bets_won} > bets_placed {bets_placed}"

        if bets_placed > 0:
            expected_sr = round(bets_won / bets_placed * 100, 1)
            assert strike_rate == expected_sr, f"{strat_key}: strike_rate {strike_rate} != {expected_sr}"

        if staked > 0:
            expected_roi = round(profit / staked * 100, 1)
            assert roi_pct == expected_roi, f"{strat_key}: roi_pct {roi_pct} != {expected_roi}"

    # Verify each month in breakdown
    for m in data["monthly_breakdown"]:
        assert m["profit"] == m["payout"] - m["staked"]
        assert m["bets_won"] <= m["bets_placed"]
        if m["bets_placed"] > 0:
            assert m["strike_rate"] == round(m["bets_won"] / m["bets_placed"] * 100, 1)


def test_brutal_cache_invalidation(client):
    """Verify cache gets cleared and re-evaluated properly."""
    # Warm up cache
    res1 = client.get("/api/portfolio/all-time")
    assert res1.status_code == 200

    # Ensure key is in cache
    assert len(_all_time_returns_cache) > 0

    # Clear cache
    _all_time_returns_cache.clear()
    assert len(_all_time_returns_cache) == 0

    # Re-fetch
    res2 = client.get("/api/portfolio/all-time")
    assert res2.status_code == 200
    assert res2.json()["total_race_days"] == res1.json()["total_race_days"]
