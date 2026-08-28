"""
Integration tests for the /api/races/date/{date_str}/predict batch prediction endpoint.
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
def sample_race_date():
    """Finds a race date in the database with multiple races."""
    with get_session() as s:
        row = s.execute(text("""
            SELECT date, COUNT(*) as c
            FROM races
            GROUP BY date
            HAVING c >= 2
            ORDER BY date DESC
            LIMIT 1
        """)).fetchone()
        return str(row[0]) if row else None


class TestBatchPredictAPI:
    def test_predict_nonexistent_date(self, client):
        response = client.post("/api/races/date/1900-01-01/predict")
        assert response.status_code == 404
        assert "No races found in database" in response.json()["detail"]

    def test_batch_predict_date_success(self, client, sample_race_date):
        if sample_race_date is None:
            pytest.skip("No races in local test database")

        response = client.post(f"/api/races/date/{sample_race_date}/predict")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert data["date"] == sample_race_date
        assert "races_predicted" in data
        assert data["races_predicted"] >= 2
        assert "entries_predicted" in data
        assert data["entries_predicted"] > 0
        assert "value_bets_count" in data
        assert "races_with_value_bets" in data
        assert isinstance(data["races_with_value_bets"], list)

        # Verify database has predictions written for this date
        with get_session() as s:
            pred_count = s.execute(text("""
                SELECT COUNT(p.id)
                FROM predictions p
                JOIN races r ON r.id = p.race_id
                WHERE r.date = :d
            """), {"d": sample_race_date}).scalar()
            assert pred_count > 0
