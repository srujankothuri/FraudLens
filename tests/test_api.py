"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def fraud_transaction():
    """A transaction that should be flagged as fraud."""
    return {
        "step": 2,
        "type": "TRANSFER",
        "amount": 350000,
        "oldbalanceOrg": 350000,
        "newbalanceOrig": 0,
        "oldbalanceDest": 0,
        "newbalanceDest": 350000,
    }


@pytest.fixture
def legit_transaction():
    """A transaction that should be classified as legit."""
    return {
        "step": 200,
        "type": "CASH_OUT",
        "amount": 300,
        "oldbalanceOrg": 15000,
        "newbalanceOrig": 14700,
        "oldbalanceDest": 80000,
        "newbalanceDest": 80300,
    }


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_service_name(self, client):
        data = client.get("/").json()
        assert data["service"] == "FraudLens API"


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_model_is_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    def test_fraud_detected(self, client, fraud_transaction):
        response = client.post("/predict", json=fraud_transaction)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 1
        assert data["prediction_label"] == "FRAUD"
        assert data["fraud_probability"] > 0.7
        assert data["risk_level"] == "HIGH"

    def test_legit_detected(self, client, legit_transaction):
        response = client.post("/predict", json=legit_transaction)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0
        assert data["prediction_label"] == "LEGIT"
        assert data["fraud_probability"] < 0.3

    def test_response_has_shap_values(self, client, fraud_transaction):
        response = client.post("/predict", json=fraud_transaction)
        data = response.json()
        assert "shap_values" in data
        assert len(data["shap_values"]) > 0

    def test_response_has_top_reasons(self, client, fraud_transaction):
        response = client.post("/predict", json=fraud_transaction)
        data = response.json()
        assert "top_reasons" in data
        assert len(data["top_reasons"]) == 5
        # Each reason should have required fields
        reason = data["top_reasons"][0]
        assert "feature" in reason
        assert "shap_value" in reason
        assert "impact" in reason

    def test_response_has_base_value(self, client, fraud_transaction):
        response = client.post("/predict", json=fraud_transaction)
        data = response.json()
        assert "base_value" in data
        assert isinstance(data["base_value"], float)

    def test_invalid_type_returns_400(self, client):
        bad_txn = {
            "step": 1,
            "type": "PAYMENT",
            "amount": 100,
            "oldbalanceOrg": 1000,
            "newbalanceOrig": 900,
            "oldbalanceDest": 5000,
            "newbalanceDest": 5100,
        }
        response = client.post("/predict", json=bad_txn)
        assert response.status_code == 400

    def test_missing_fields_returns_422(self, client):
        response = client.post("/predict", json={"amount": 100})
        assert response.status_code == 422

    def test_negative_amount_returns_422(self, client):
        bad_txn = {
            "step": 1,
            "type": "TRANSFER",
            "amount": -500,
            "oldbalanceOrg": 1000,
            "newbalanceOrig": 1500,
            "oldbalanceDest": 0,
            "newbalanceDest": 0,
        }
        response = client.post("/predict", json=bad_txn)
        assert response.status_code == 422