"""
Integration tests for the /api/scraper/date/{date_str} endpoint and refetch logic.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestScraperAPI:
    def test_scrape_date_endpoint(self, client):
        with patch("pipeline.step_scrape", return_value=["202608290101", "202608290102"]) as mock_scrape:
            response = client.post("/api/scraper/date/2026-08-29")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["races_count"] == 2
            assert data["race_ids"] == ["202608290101", "202608290102"]
            mock_scrape.assert_called_once_with("2026-08-29", force=False)

    def test_scrape_date_force_endpoint(self, client):
        with patch("pipeline.step_scrape", return_value=["202608290101"]) as mock_scrape:
            response = client.post("/api/scraper/date/2026-08-29?force=true")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["races_count"] == 1
            mock_scrape.assert_called_once_with("2026-08-29", force=True)

    def test_sync_results_endpoint(self, client):
        with patch("pipeline.step_results", return_value=[101, 102]) as mock_results:
            response = client.post("/api/results/date/2026-08-23")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["races_synced"] == 2
            assert data["race_ids"] == [101, 102]
            mock_results.assert_called_once_with("2026-08-23", force=False)

