"""
Unit tests for Phase 12: Real-Time Fraud & Anomaly Detection, VIN Decoder & Dealer Analytics Suite.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app
from src.fraud_detector import ListingFraudDetector
from src.vin_decoder import VINDecoder
from src.dealer_analytics import DealerAnalyticsEngine


client = TestClient(app)


def test_fraud_detector_normal():
    """Test fraud detector on normal listing."""
    detector = ListingFraudDetector()
    car_data = {
        "manufacture_year": 2020,
        "km_driven": 35000,
        "owner_count": 1,
        "asking_price": 550000,
        "accident_history": "No"
    }

    res = detector.evaluate_listing_fraud(car_data, predicted_price=550000)
    assert res["fraud_risk_score"] < 0.25
    assert res["risk_level"] == "LOW"
    assert res["is_safe_listing"] == True


def test_fraud_detector_anomaly():
    """Test fraud detector on odometer tampering and underpriced anomaly listing."""
    detector = ListingFraudDetector()
    car_data = {
        "manufacture_year": 2016, # 10 years old
        "km_driven": 2000,        # Suspiciously low 2,000 km
        "owner_count": 4,         # Frequent transfers
        "asking_price": 100000,   # > 50% underpriced
        "accident_history": "Yes"
    }

    res = detector.evaluate_listing_fraud(car_data, predicted_price=400000)
    assert res["fraud_risk_score"] >= 0.50
    assert res["risk_level"] == "HIGH"
    assert len(res["anomaly_flags"]) >= 3
    assert res["is_safe_listing"] == False


def test_vin_decoder():
    """Test 17-character VIN decoder."""
    decoder = VINDecoder()

    # Valid Maruti VIN
    vin_maruti = "MA3FCEB1S00123456"
    res_m = decoder.decode_vin(vin_maruti)
    assert res_m["valid"] == True
    assert res_m["company_name"] == "Maruti"

    # Valid Hyundai VIN
    vin_hyundai = "MBHM21P0009876543"
    res_h = decoder.decode_vin(vin_hyundai)
    assert res_h["valid"] == True
    assert res_h["company_name"] == "Hyundai"

    # Invalid VIN length
    res_inv = decoder.decode_vin("INVALID123")
    assert res_inv["valid"] == False


def test_dealer_analytics_engine():
    """Test commercial dealer pricing tiers and 5-year depreciation curve."""
    engine = DealerAnalyticsEngine()
    analytics = engine.generate_dealer_analytics(fair_market_price=600000.0, manufacture_year=2020)

    assert "pricing_tiers" in analytics
    tiers = analytics["pricing_tiers"]
    assert tiers["trade_in_wholesale"] < tiers["private_party"] < tiers["retail_showroom"]

    assert analytics["dealer_gross_margin_inr"] > 0
    assert len(analytics["depreciation_projections"]) == 5
    assert analytics["5_year_future_val"] < analytics["1_year_future_val"]


def test_fraud_endpoint():
    """Test POST /fraud/evaluate endpoint."""
    response = client.post("/fraud/evaluate", json={
        "manufacture_year": 2021,
        "km_driven": 25000,
        "owner_count": 1,
        "asking_price": 500000
    })
    assert response.status_code == 200
    data = response.json()
    assert "fraud_risk_score" in data
    assert "risk_level" in data


def test_vin_endpoint():
    """Test GET /vin/decode/{vin} endpoint."""
    response = client.get("/vin/decode/MA3FCEB1S00123456")
    assert response.status_code == 200
    data = response.json()
    assert data["company_name"] == "Maruti"
    assert data["valid"] == True


def test_dealer_analytics_endpoint():
    """Test POST /dealer/analytics endpoint."""
    response = client.post("/dealer/analytics", json={
        "company_name": "Honda",
        "model_name": "City",
        "manufacture_year": 2021,
        "km_driven": 30000
    })
    assert response.status_code == 200
    data = response.json()
    assert "pricing_tiers" in data
    assert "depreciation_projections" in data
