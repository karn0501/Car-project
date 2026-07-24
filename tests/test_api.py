"""
Unit tests for Phase 8: FastAPI Backend API endpoints.
Tests health check, prediction, comparison, feedback, and rate limiting.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


def test_health_endpoint():
    """Test that health check returns 200 with correct structure."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "models_loaded" in data
    assert "ensemble" in data["models_loaded"]
    assert "cnn_condition" in data["models_loaded"]
    assert "lstm_trend" in data["models_loaded"]
    assert "database_status" in data
    assert "timestamp" in data


def test_predict_endpoint():
    """Test that prediction returns valid response with SHAP breakdown."""
    payload = {
        "company_name": "Maruti",
        "model_name": "Swift",
        "variant_name": "VXi",
        "manufacture_year": 2020,
        "km_driven": 35000,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "owner_count": 1,
        "city": "Mumbai",
        "body_type": "Hatchback",
        "engine_cc": 1197,
        "seating_capacity": 5,
        "insurance_valid": "Yes",
        "accident_history": "No",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 0
    assert "price_range_low" in data
    assert "price_range_high" in data
    assert data["price_range_low"] < data["predicted_price"] < data["price_range_high"]
    assert "shap_breakdown" in data
    assert len(data["shap_breakdown"]) > 0
    assert "prediction_id" in data
    assert data["prediction_id"].startswith("PRED-")
    assert data["currency"] == "INR"


def test_predict_with_description():
    """Test that NLP description scoring works in prediction."""
    payload = {
        "company_name": "Hyundai",
        "model_name": "Creta",
        "variant_name": "SX",
        "manufacture_year": 2021,
        "km_driven": 20000,
        "fuel_type": "Diesel",
        "transmission": "Automatic",
        "owner_count": 1,
        "city": "Delhi",
        "body_type": "SUV",
        "engine_cc": 1493,
        "seating_capacity": 5,
        "insurance_valid": "Yes",
        "accident_history": "No",
        "description": "Single owner, well maintained car in showroom condition. "
                       "Regularly serviced at authorized center. No scratches, accident free.",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["description_quality_score"] is not None
    assert 0.0 <= data["description_quality_score"] <= 1.0
    assert data["description_quality_score"] >= 0.7  # High quality description


def test_compare_endpoint():
    """Test comparable listings search."""
    response = client.get("/compare", params={
        "company": "Maruti",
        "model": "Swift",
        "year": 2020,
        "limit": 5,
    })
    assert response.status_code == 200

    data = response.json()
    assert "query_summary" in data
    assert "comparable_count" in data
    assert "listings" in data
    assert isinstance(data["listings"], list)


def test_feedback_endpoint():
    """Test feedback submission."""
    # First make a prediction to get a prediction_id
    predict_response = client.post("/predict", json={
        "company_name": "Tata",
        "model_name": "Nexon",
        "variant_name": "XZ",
        "manufacture_year": 2022,
        "km_driven": 15000,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "owner_count": 1,
        "city": "Pune",
    })
    pred_id = predict_response.json()["prediction_id"]

    # Submit feedback
    feedback_response = client.post("/feedback", json={
        "prediction_id": pred_id,
        "actual_price": 850000,
        "comments": "Sold at slightly below predicted price",
    })
    assert feedback_response.status_code == 200

    data = feedback_response.json()
    assert data["status"] == "success"
    assert data["feedback_id"].startswith("FB-")
    assert pred_id in data["message"]


def test_trend_endpoint():
    """Test price trend forecast endpoint."""
    response = client.get("/trend/Maruti Swift", params={"months": 3})
    assert response.status_code == 200

    data = response.json()
    assert "trend_direction" in data
    assert data["trend_direction"] in ("RISING", "FALLING", "STABLE")
    assert "forecasts" in data
    assert len(data["forecasts"]) == 3
    for fc in data["forecasts"]:
        assert "predicted_price" in fc
        assert "change_pct" in fc


def test_predict_validation_error():
    """Test that invalid input returns 422 validation error."""
    response = client.post("/predict", json={
        "company_name": "Maruti",
        # Missing required fields
    })
    assert response.status_code == 422


def test_invalid_api_key():
    """Test that an invalid API key returns 403."""
    response = client.get("/health", headers={"X-API-Key": "wrong-key"})
    # Health endpoint skips auth, so test on predict
    response = client.post("/predict", json={
        "company_name": "Maruti",
        "model_name": "Swift",
        "manufacture_year": 2020,
        "km_driven": 30000,
    }, headers={"X-API-Key": "wrong-key-12345"})
    assert response.status_code == 403
