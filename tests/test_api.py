from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import api


class FakeService:
    def __init__(self):
        self.first_date = pd.Timestamp("2012-11-01")
        self.last_date = pd.Timestamp("2012-11-02")
        self.artifact = {"model_type": "weekly_baseline"}
        self.calls = []

        hours = pd.date_range(self.first_date, periods=24, freq="h")
        self.history = pd.DataFrame({"cnt": 10.0}, index=hours)
        self.history.iloc[3, 0] = float("nan")

    def forecast(self, forecast_date):
        self.calls.append(forecast_date)
        hours = pd.date_range(forecast_date, periods=24, freq="h")
        return {
            "forecast_date": forecast_date.isoformat(),
            "model_type": "weekly_baseline",
            "model_sha256": "synthetic-test-artifact",
            "nominal_interval_coverage": 0.9,
            "forecast": [
                {
                    "timestamp": timestamp.isoformat(),
                    "prediction": 10.0,
                    "lower": 0.0,
                    "upper": 30.0,
                    "prediction_source": "same_hour_last_week",
                }
                for timestamp in hours
            ],
            "note": "Synthetic API test fixture.",
        }


@pytest.fixture
def client_and_service(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(api, "ForecastService", lambda: service)
    with TestClient(api.app) as client:
        yield client, service


def test_health(client_and_service):
    client, _ = client_and_service
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["first_forecast_date"] == "2012-11-01"


def test_forecast_returns_24_hours(client_and_service):
    client, service = client_and_service
    response = client.post(
        "/forecast", json={"forecast_date": "2012-11-01"}
    )
    assert response.status_code == 200
    assert service.calls == [date(2012, 11, 1)]
    rows = response.json()["forecast"]
    assert len(rows) == 24
    assert rows[0]["timestamp"] == "2012-11-01T00:00:00"
    assert rows[-1]["timestamp"] == "2012-11-01T23:00:00"


@pytest.mark.parametrize("payload", [
    {},
    {"forecast_date": "not-a-date"},
    {"forecast_date": "2012-10-31"},
    {"forecast_date": "2012-11-03"},
    {"forecast_date": "2012-11-01", "unexpected": True},
])
def test_invalid_request_does_not_call_forecaster(
    client_and_service, payload
):
    client, service = client_and_service
    response = client.post("/forecast", json=payload)
    assert response.status_code == 422
    assert service.calls == []


def test_actuals_preserve_missing_values(client_and_service):
    client, service = client_and_service
    response = client.get(
        "/actuals", params={"forecast_date": "2012-11-01"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["observed_hours"] == 23
    assert len(body["actuals"]) == 24
    assert body["actuals"][3]["actual"] is None
    assert service.calls == []


def test_day_without_actuals_returns_nulls(client_and_service):
    client, _ = client_and_service
    response = client.get(
        "/actuals", params={"forecast_date": "2012-11-02"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["observed_hours"] == 0
    assert all(row["actual"] is None for row in body["actuals"])


def test_actuals_reject_out_of_range_date(client_and_service):
    client, _ = client_and_service
    response = client.get(
        "/actuals", params={"forecast_date": "2026-01-01"}
    )
    assert response.status_code == 422


def test_demo_page_is_served(client_and_service):
    client, _ = client_and_service
    response = client.get("/demo")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Demand Forecast Lab" in response.text
