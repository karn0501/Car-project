"""
Unit tests for Phase 9: Market Growth Features, AI Conversational Chatbot, JWT Auth & Web UI.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app
from api.nlp_query_parser import CarQueryParser


client = TestClient(app)


def test_natural_language_query_parser():
    """Test entity extraction from free-text vehicle queries."""
    parser = CarQueryParser()

    # Query 1
    q1 = "2021 Hyundai Creta SX Diesel automatic 25,000 km in Delhi"
    parsed1 = parser.parse_query(q1)

    assert parsed1["company_name"] == "Hyundai"
    assert parsed1["model_name"] == "Creta"
    assert parsed1["manufacture_year"] == 2021
    assert parsed1["km_driven"] == 25000.0
    assert parsed1["fuel_type"] == "Diesel"
    assert parsed1["transmission"] == "Automatic"
    assert parsed1["city"] == "Delhi"

    # Query 2
    q2 = "2019 Maruti Swift VXi petrol 40000 km in Mumbai"
    parsed2 = parser.parse_query(q2)

    assert parsed2["company_name"] == "Maruti"
    assert parsed2["model_name"] == "Swift"
    assert parsed2["manufacture_year"] == 2019
    assert parsed2["km_driven"] == 40000.0
    assert parsed2["fuel_type"] == "Petrol"
    assert parsed2["city"] == "Mumbai"


def test_auth_registration_and_login():
    """Test user account registration and login JWT issue."""
    email = f"testuser_{os.urandom(4).hex()}@example.com"
    password = "SecurePassword123"

    # 1. Register
    reg_resp = client.post("/auth/register", json={
        "email": email,
        "password": password,
        "full_name": "Test User"
    })
    assert reg_resp.status_code == 200
    reg_data = reg_resp.json()
    assert "access_token" in reg_data
    assert reg_data["user_email"] == email

    # 2. Login
    login_resp = client.post("/auth/login", json={
        "email": email,
        "password": password
    })
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]

    # 3. Access Protected /auth/me
    profile_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert profile_resp.status_code == 200
    profile_data = profile_resp.json()
    assert profile_data["email"] == email
    assert profile_data["full_name"] == "Test User"


def test_chat_predict_endpoint():
    """Test natural language chat-predict endpoint."""
    response = client.post("/chat-predict", json={
        "query": "2020 Honda City VX CVT petrol 30000 km in Bangalore"
    })
    assert response.status_code == 200

    data = response.json()
    assert "query" in data
    assert "parsed" in data
    assert "prediction" in data

    parsed = data["parsed"]
    assert parsed["company_name"] == "Honda"
    assert parsed["model_name"] == "City"
    assert parsed["manufacture_year"] == 2020

    pred = data["prediction"]
    assert pred["predicted_price"] > 0
    assert "shap_breakdown" in pred


def test_root_ui_dashboard_endpoint():
    """Test that root / serves index.html dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "AutoValuate" in response.text
