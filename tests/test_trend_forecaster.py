"""
Unit tests for Phase 7: LSTM/GRU Price Trend Forecaster.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.trend_forecaster import (
    PriceTrendLSTM,
    PriceTrendForecaster,
    get_fuel_price_index,
    get_interest_rate_index,
    get_festive_season_marker,
)
from datetime import datetime


def test_macroeconomic_signal_ranges():
    """Test that all macroeconomic signal generators return values in [0, 1]."""
    test_dates = [
        datetime(2024, 1, 15),
        datetime(2024, 6, 20),
        datetime(2024, 10, 5),
        datetime(2025, 3, 1),
        datetime(2025, 11, 15),
    ]

    for date in test_dates:
        fuel = get_fuel_price_index(date)
        interest = get_interest_rate_index(date)
        festive = get_festive_season_marker(date)

        assert 0.0 <= fuel <= 1.0, f"Fuel index out of range: {fuel}"
        assert 0.0 <= interest <= 1.0, f"Interest index out of range: {interest}"
        assert festive in (0.0, 1.0), f"Festive marker should be 0 or 1: {festive}"

    # October/November should be festive
    assert get_festive_season_marker(datetime(2024, 10, 1)) == 1.0
    assert get_festive_season_marker(datetime(2024, 11, 1)) == 1.0
    # June should not be festive
    assert get_festive_season_marker(datetime(2024, 6, 1)) == 0.0


def test_lstm_forward_pass():
    """Test that LSTM model produces correct output shapes."""
    model = PriceTrendLSTM(input_size=6, hidden_size=64, num_layers=2, output_size=1)

    # Batch of 4 sequences, each 6 timesteps, 6 features
    dummy_input = torch.randn(4, 6, 6)
    output = model(dummy_input)

    assert output.shape == (4, 1), f"Expected (4,1) output, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"


def test_price_history_generation():
    """Test synthetic price history generation from listings data."""
    df_test = pd.DataFrame([
        {"company_name": "Maruti", "model_name": "Swift", "city": "Mumbai",
         "asking_price": 500000, "manufacture_year": 2020},
        {"company_name": "Maruti", "model_name": "Swift", "city": "Mumbai",
         "asking_price": 520000, "manufacture_year": 2020},
        {"company_name": "Maruti", "model_name": "Swift", "city": "Mumbai",
         "asking_price": 480000, "manufacture_year": 2020},
        {"company_name": "Hyundai", "model_name": "Creta", "city": "Delhi",
         "asking_price": 1200000, "manufacture_year": 2021},
        {"company_name": "Hyundai", "model_name": "Creta", "city": "Delhi",
         "asking_price": 1150000, "manufacture_year": 2021},
        {"company_name": "Hyundai", "model_name": "Creta", "city": "Delhi",
         "asking_price": 1250000, "manufacture_year": 2021},
    ])

    forecaster = PriceTrendForecaster(model_path="models/test_trend.pt", seq_length=6)
    history = forecaster.generate_price_history(df_test, months=12)

    assert len(history) > 0, "Should generate price history records"
    assert "avg_price" in history.columns
    assert "fuel_index" in history.columns
    assert "interest_index" in history.columns
    assert "festive_marker" in history.columns
    assert "month_sin" in history.columns
    assert "month_cos" in history.columns
    assert all(history["avg_price"] > 0), "All prices should be positive"


def test_forecast_output_shape():
    """Test that forecast produces correct output format."""
    forecaster = PriceTrendForecaster(model_path="models/test_trend.pt", seq_length=4)

    # Create minimal test history
    df_test = pd.DataFrame([
        {"company_name": "Maruti", "model_name": "Swift", "city": "Mumbai",
         "asking_price": 500000, "manufacture_year": 2020},
        {"company_name": "Maruti", "model_name": "Swift", "city": "Mumbai",
         "asking_price": 490000, "manufacture_year": 2020},
        {"company_name": "Maruti", "model_name": "Swift", "city": "Mumbai",
         "asking_price": 510000, "manufacture_year": 2020},
    ])

    history = forecaster.generate_price_history(df_test, months=12)

    if len(history) >= forecaster.seq_length:
        # Train briefly
        forecaster.train(history, epochs=5, lr=0.01)

        # Forecast
        sample = history[history["variant_key"] == history["variant_key"].iloc[0]].tail(12)
        result = forecaster.forecast(sample, months_ahead=3)

        assert result["status"] == "success"
        assert "trend_direction" in result
        assert result["trend_direction"] in ("RISING", "FALLING", "STABLE")
        assert len(result["forecasts"]) == 3
        for fc in result["forecasts"]:
            assert "predicted_price" in fc
            assert "change_pct" in fc
            assert fc["predicted_price"] > 0
