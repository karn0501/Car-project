"""
Unit tests for Phase 2: Data Preprocessing, Feature Engineering, and Baseline Model Training.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.data_exporter import load_dataset_from_db
from src.preprocessing import DataPreprocessor
from src.baseline_model import BaselineCatBoostModel


def test_data_exporter():
    df = load_dataset_from_db()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "asking_price" in df.columns
    assert "company_name" in df.columns
    assert "km_driven" in df.columns


def test_feature_engineering():
    preprocessor = DataPreprocessor(current_year=2026)
    df_raw = pd.DataFrame([{
        "company_name": "Maruti Suzuki",
        "model_name": "Swift",
        "variant_name": "VXI",
        "body_type": "Hatchback",
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "city": "Delhi",
        "manufacture_year": 2021,
        "km_driven": 25000.0,
        "owner_count": 1,
        "engine_cc": 1197,
        "seating_capacity": 5,
        "ex_showroom_price": 729000.0,
        "asking_price": 550000.0,
        "model_discontinued_year": None,
        "insurance_valid": True,
        "accident_history": False
    }])

    df_fe = preprocessor.engineer_features(df_raw)
    assert "car_age" in df_fe.columns
    assert df_fe["car_age"].iloc[0] == 5  # 2026 - 2021
    assert "price_per_km" in df_fe.columns
    assert df_fe["price_per_km"].iloc[0] == pytest.approx(550000.0 / 25000.0)
    assert "depreciation_ratio" in df_fe.columns
    assert df_fe["depreciation_ratio"].iloc[0] == pytest.approx(550000.0 / 729000.0)


def test_baseline_model_training(tmp_path):
    df_raw = load_dataset_from_db()
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, cat_features = preprocessor.prepare_splits(df_raw)

    model_wrapper = BaselineCatBoostModel(model_dir=str(tmp_path))
    model_wrapper.fit(X_train, y_train, cat_features=cat_features)
    metrics = model_wrapper.evaluate(X_test, y_test)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2_score" in metrics
    assert metrics["r2_score"] > 0.0  # Positive R2 score

    # Test prediction function
    preds = model_wrapper.predict(X_test.iloc[:5])
    assert len(preds) == 5
    assert all(p > 0 for p in preds)
