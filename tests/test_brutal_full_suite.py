"""
Brutal comprehensive test suite for UmaEdge FastAPI backend.
Tests all endpoints, query filter permutations, betting analysis strategies,
caching layers, and SQLite concurrency under stress.
"""

import pytest
import concurrent.futures
from fastapi.testclient import TestClient
from api.main import app, _daily_returns_cache
from scraper.db import get_session
from sqlalchemy import text


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


class TestManifestEndpoint:
    def test_manifest_success(self, client):
        res = client.get("/api/manifest")
        assert res.status_code == 200
        data = res.json()
        assert "dates" in data
        assert isinstance(data["dates"], list)
        assert len(data["dates"]) > 0
        for item in data["dates"]:
            assert "date" in item
            assert "races" in item
            assert "bets" in item
            assert "winners" in item
            assert "strike_rate" in item


class TestRacesEndpointFiltering:
    def test_list_races_all_date_filters(self, client):
        # 1. Date filter
        res = client.get("/api/races?date=2026-08-29&limit=100")
        assert res.status_code == 200
        data = res.json()
        assert data["count"] > 0
        for r in data["races"]:
            assert r["date"] == "2026-08-29"
            assert "has_predictions" in r
            assert "has_bet" in r

    def test_list_races_course_filters(self, client):
        courses = ["Sapporo", "Niigata", "Chukyo"]
        for c in courses:
            res = client.get(f"/api/races?date=2026-08-29&course={c}&limit=100")
            assert res.status_code == 200
            data = res.json()
            assert data["count"] > 0
            for r in data["races"]:
                assert r["course_name"] == c
                assert r["date"] == "2026-08-29"

    def test_list_races_pagination(self, client):
        res1 = client.get("/api/races?date=2026-08-29&limit=5&offset=0")
        res2 = client.get("/api/races?date=2026-08-29&limit=5&offset=5")
        assert res1.status_code == 200
        assert res2.status_code == 200
        ids1 = [r["id"] for r in res1.json()["races"]]
        ids2 = [r["id"] for r in res2.json()["races"]]
        assert len(ids1) == 5
        assert len(ids2) == 5
        assert set(ids1).isdisjoint(set(ids2))

    def test_list_races_empty_date(self, client):
        res = client.get("/api/races?date=1970-01-01")
        assert res.status_code == 200
        assert res.json()["count"] == 0
        assert res.json()["races"] == []


class TestRaceDetailEndpoint:
    def test_get_race_detail_all_2026_08_29(self, client):
        # Fetch list of race IDs
        list_res = client.get("/api/races?date=2026-08-29&limit=100")
        assert list_res.status_code == 200
        races = list_res.json()["races"]
        assert len(races) >= 36

        # Check first 5 races detail
        for r in races[:5]:
            rid = r["id"]
            res = client.get(f"/api/races/{rid}")
            assert res.status_code == 200
            detail = res.json()
            assert "race" in detail
            assert "entries" in detail
            assert "predictions" in detail
            assert "value_bets" in detail
            assert detail["race"]["id"] == rid
            assert len(detail["entries"]) > 0

    def test_get_race_detail_not_found(self, client):
        res = client.get("/api/races/999999999")
        assert res.status_code == 404


class TestBettingAnalysisEndpoint:
    def test_all_strategies_and_budgets(self, client):
        # Find a race with predictions
        list_res = client.get("/api/races?date=2026-08-29&limit=100")
        races = [r for r in list_res.json()["races"] if r.get("has_predictions")]
        assert len(races) > 0
        race_id = races[0]["id"]

        budgets = [500, 1000, 2500, 5000, 10000]
        modes = ["auto", "hybrid", "pure_win", "dutching"]

        for b in budgets:
            for m in modes:
                res = client.get(f"/api/races/{race_id}/betting-analysis?budget={b}&mode={m}")
                assert res.status_code == 200
                data = res.json()
                assert "headline" in data
                assert "staking_plan" in data
                assert "pricing_table" in data
                assert "strategy_result" in data
                assert "all_strategies_results" in data
                assert data["staking_plan"]["budget"] == b


class TestDailyReturnsEndpoint:
    def test_daily_returns_calculation_and_caching(self, client):
        _daily_returns_cache.clear()
        date_str = "2026-08-29"

        # 1. First call (uncached)
        res1 = client.get(f"/api/races/date/{date_str}/daily-returns?budget_per_race=1000")
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["date"] == date_str
        assert "strategies" in data1
        assert "auto" in data1["strategies"]

        # 2. Second call (cached)
        res2 = client.get(f"/api/races/date/{date_str}/daily-returns?budget_per_race=1000")
        assert res2.status_code == 200
        assert res2.json() == data1

        # 3. Force refresh
        res3 = client.get(f"/api/races/date/{date_str}/daily-returns?budget_per_race=1000&force_refresh=true")
        assert res3.status_code == 200
        assert res3.json()["date"] == date_str


class TestConcurrentLoadStress:
    def test_concurrent_readers(self, client):
        """Spawns 20 concurrent readers across multiple endpoints to test SQLite WAL concurrency."""
        urls = [
            "/api/manifest",
            "/api/races?date=2026-08-29&limit=100",
            "/api/races?date=2026-08-29&course=Sapporo",
            "/api/races/date/2026-08-29/daily-returns?budget_per_race=1000",
            "/api/races/43707",
            "/api/races/43708",
            "/api/races/43709",
            "/api/races/43707/betting-analysis?budget=1000&mode=auto",
            "/api/races/43708/betting-analysis?budget=5000&mode=pure_win",
            "/api/races/43709/betting-analysis?budget=3000&mode=dutching",
        ] * 4  # 40 requests

        def _fetch(u):
            r = client.get(u)
            return (u, r.status_code)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            results = list(ex.map(_fetch, urls))

        for url, status in results:
            assert status == 200, f"URL {url} returned status {status}"


class TestMonthlyReturnsEndpoint:
    def test_monthly_returns_success(self, client):
        """Test monthly returns aggregation for 2026-08 across all 4 modes."""
        res = client.get("/api/races/month/2026-08/monthly-returns?budget_per_race=1000")
        assert res.status_code == 200
        data = res.json()
        assert data["month"] == "2026-08"
        assert "strategies" in data
        assert "auto" in data["strategies"]
        assert "pure_win" in data["strategies"]
        assert "hybrid" in data["strategies"]
        assert "dutching" in data["strategies"]
        assert "daily_timeline" in data
        assert isinstance(data["daily_timeline"], list)
        assert len(data["daily_timeline"]) > 0

    def test_monthly_returns_invalid_month(self, client):
        """Test invalid month format returns 400."""
        res = client.get("/api/races/month/invalid-month/monthly-returns")
        assert res.status_code == 400
