"""
Unit tests for Phase 10: Enterprise Ecosystem, Model Drift Monitoring & B2B Extensions.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app
from src.drift_detector import ModelDriftDetector
from api.metering import APIMeteringManager
from scripts.benchmark_api import run_benchmark


client = TestClient(app)


def test_drift_detector_healthy():
    """Test data drift detector on similar distributions."""
    detector = ModelDriftDetector()
    df_live = detector.baseline_df.copy()

    report = detector.evaluate_dataset_health(df_live)
    assert report["overall_status"] == "HEALTHY"
    assert report["drifted_feature_count"] == 0


def test_drift_detector_anomalous():
    """Test data drift detector on shifted distribution (out-of-distribution data)."""
    detector = ModelDriftDetector()

    # Shift km_driven by 500,000 km to trigger drift
    df_shifted = detector.baseline_df.copy()
    df_shifted["km_driven"] = df_shifted["km_driven"] + 500000.0

    report = detector.evaluate_dataset_health(df_shifted)
    assert report["overall_status"] == "DRIFT_DETECTED"
    assert report["drifted_feature_count"] >= 1


def test_b2b_metering_manager():
    """Test API key validation and quota tracking."""
    meter = APIMeteringManager()

    # Valid key
    res = meter.validate_and_record("car-prediction-api-key-2026")
    assert res["valid"] == True
    assert res["tier"] == "FREE"

    # Metrics lookup
    metrics = meter.get_key_metrics("car-prediction-api-key-2026")
    assert metrics is not None
    assert metrics["tier"] == "FREE"
    assert metrics["total_lifetime_requests"] >= 1

    # Invalid key
    res_invalid = meter.validate_and_record("invalid-key-999999")
    assert res_invalid["valid"] == False


def test_prometheus_metrics_endpoint():
    """Test Prometheus metrics endpoint output format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "car_predictions_total" in response.text
    assert "car_api_status 1" in response.text


def test_api_usage_endpoint():
    """Test /api/usage endpoint."""
    response = client.get("/api/usage", headers={"X-API-Key": "car-prediction-api-key-2026"})
    assert response.status_code == 200
    data = response.json()
    assert data["tier"] == "FREE"
    assert "limit_per_minute" in data


def test_drift_check_endpoint():
    """Test POST /drift/check endpoint."""
    response = client.post("/drift/check")
    assert response.status_code == 200
    data = response.json()
    assert "overall_status" in data
    assert "checked_feature_count" in data


def test_api_benchmark_execution():
    """Test load-testing benchmark runner."""
    res = run_benchmark(num_requests=10, max_workers=2)
    assert res["rps"] > 0
    assert res["success_rate"] == 100.0
    assert res["p50_ms"] > 0
