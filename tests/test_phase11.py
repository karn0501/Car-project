"""
Unit tests for Phase 11: Multi-Region Localization, Dynamic Currency Engine & i18n Translation.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app
from src.currency_engine import CurrencyConverter
from src.regional_tax import RegionalTaxCalculator
from src.i18n import TranslationEngine


client = TestClient(app)


def test_currency_conversion():
    """Test dynamic currency conversion and formatting across currencies."""
    converter = CurrencyConverter()

    # INR conversion
    assert converter.convert_from_inr(100000.0, "INR") == 100000.0
    assert "Lakhs" in converter.format_price(650000.0, "INR")

    # USD conversion
    usd_val = converter.convert_from_inr(650000.0, "USD")
    assert usd_val == 7800.0
    assert "$" in converter.format_price(usd_val, "USD")

    # EUR conversion
    eur_val = converter.convert_from_inr(650000.0, "EUR")
    assert eur_val > 0
    assert "€" in converter.format_price(eur_val, "EUR")

    # GBP conversion
    gbp_val = converter.convert_from_inr(650000.0, "GBP")
    assert gbp_val > 0
    assert "£" in converter.format_price(gbp_val, "GBP")

    # AED conversion
    aed_val = converter.convert_from_inr(650000.0, "AED")
    assert aed_val > 0
    assert "AED" in converter.format_price(aed_val, "AED")

    # JPY conversion
    jpy_val = converter.convert_from_inr(650000.0, "JPY")
    assert jpy_val > 0
    assert "¥" in converter.format_price(jpy_val, "JPY")


def test_regional_tax_calculator():
    """Test state RTO tax rates and Delhi NCR diesel penalty."""
    calculator = RegionalTaxCalculator()

    # Mumbai (12% tax)
    res_mumbai = calculator.calculate_regional_fees("Mumbai", 500000.0, "Petrol", 3)
    assert res_mumbai["rto_tax_rate_pct"] == 12.0
    assert res_mumbai["estimated_rto_tax"] == 60000.0
    assert res_mumbai["ncr_diesel_penalty"] == 0.0

    # Delhi NCR 8-year Diesel vehicle (25% penalty)
    res_delhi = calculator.calculate_regional_fees("Delhi", 600000.0, "Diesel", 8)
    assert res_delhi["ncr_diesel_penalty"] == 150000.0

    # Electric car (0% RTO tax)
    res_ev = calculator.calculate_regional_fees("Bangalore", 1000000.0, "Electric", 2)
    assert res_ev["estimated_rto_tax"] == 0.0


def test_i18n_translation():
    """Test multi-language translation for EN, HI, ES, AR."""
    translator = TranslationEngine()

    assert translator.translate_key("title", "en") == "Car Valuation Report"
    assert translator.translate_key("title", "hi") == "कार मूल्यांकन रिपोर्ट"
    assert translator.translate_key("title", "es") == "Informe de Valoración de Vehículo"
    assert translator.translate_key("title", "ar") == "تقرير تقييم السيارة"

    # SHAP translation
    shap = [{"feature": "Car Age", "impact_inr": -20000, "direction": "negative"}]
    translated = translator.translate_shap_breakdown(shap, "hi")
    assert translated[0]["feature"] == "कार की आयु"
    assert translated[0]["direction"] == "नकारात्मक"


def test_currencies_endpoint():
    """Test GET /currencies endpoint."""
    response = client.get("/currencies")
    assert response.status_code == 200
    data = response.json()
    assert "currencies" in data
    assert "languages" in data
    assert "USD" in data["currencies"]
    assert "hi" in data["languages"]


def test_localized_prediction_endpoint():
    """Test POST /predict/localized endpoint with USD currency and Hindi language."""
    payload = {
        "company_name": "Hyundai",
        "model_name": "Creta",
        "variant_name": "SX",
        "manufacture_year": 2021,
        "km_driven": 20000,
        "fuel_type": "Diesel",
        "transmission": "Automatic",
        "owner_count": 1,
        "city": "Delhi"
    }

    response = client.post("/predict/localized?currency=USD&lang=hi", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["currency"] == "USD"
    assert data["language"] == "hi"
    assert "$" in data["formatted_price"]
    assert "regional_tax_breakdown" in data
    assert "total_buyer_landed_cost" in data["regional_tax_breakdown"]
