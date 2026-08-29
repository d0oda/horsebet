import concurrent.futures
import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    return TestClient(app)

class TestAPIConcurrencyAndStress:
    """Stress tests concurrent API calls and malicious/extreme input payloads."""

    def test_concurrent_api_requests(self, client):
        endpoints = [
            "/api/races?date=2026-08-29",
            "/api/races?limit=50&offset=0",
            "/api/races/date/2026-08-29/daily-returns?budget_per_race=1000",
            "/api/races/month/2026-08/monthly-returns?budget_per_race=1000",
            "/api/health",
        ]

        def fetch(url):
            return client.get(url).status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(fetch, url) for url in endpoints * 10]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(status == 200 for status in results)
        assert len(results) == len(endpoints) * 10

    def test_malicious_and_boundary_inputs(self, client):
        # SQL Injection attempts in date and course query parameters
        res = client.get("/api/races?date=' OR 1=1 --")
        assert res.status_code in [200, 400, 404, 422]  # Should not crash or 500
        if res.status_code == 200:
            assert "races" in res.json()

        res = client.get("/api/races?course='; DROP TABLE races; --")
        assert res.status_code in [200, 400, 404, 422]

        # XSS payloads
        res = client.get("/api/races?date=<script>alert(1)</script>")
        assert res.status_code in [200, 400, 404, 422]

        # Extreme offsets and limits
        res = client.get("/api/races?limit=1000000&offset=999999999")
        assert res.status_code in [200, 422]
        if res.status_code == 200:
            assert res.json()["races"] == []

        # Negative budgets
        res = client.get("/api/races/date/2026-08-29/daily-returns?budget_per_race=-500")
        assert res.status_code == 422  # FastAPI validation error

        # Huge budgets
        res = client.get("/api/races/date/2026-08-29/daily-returns?budget_per_race=100000000000")
        assert res.status_code == 422

        # Invalid month formats
        res = client.get("/api/races/month/2026-99/monthly-returns")
        # Either 404 or 400 but never unhandled 500
        assert res.status_code in [400, 404]

    def test_cache_concurrency_race_condition(self, client):
        # Test simultaneous force_refresh cache clears and readers
        def do_refresh():
            return client.get("/api/races/date/2026-08-29/daily-returns?force_refresh=true").status_code

        def do_read():
            return client.get("/api/races/date/2026-08-29/daily-returns").status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            f_refreshes = [executor.submit(do_refresh) for _ in range(5)]
            f_reads = [executor.submit(do_read) for _ in range(15)]
            all_statuses = [f.result() for f in concurrent.futures.as_completed(f_refreshes + f_reads)]

        assert all(s == 200 for s in all_statuses)
