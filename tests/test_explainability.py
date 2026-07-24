"""
Unit tests for Phase 4: Explainable AI (SHAP TreeExplainer & Price Breakdown Engine).
"""

import os
import sys
import pytest
import pandas as pd

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.data_exporter import load_dataset_from_db
from src.preprocessing import DataPreprocessor
from src.explainability import CarPriceExplainer


def test_explainability_instance_breakdown():
    df_raw = load_dataset_from_db()
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, cat_features = preprocessor.prepare_splits(df_raw)

    explainer = CarPriceExplainer()
    single_car = X_test.iloc[[0]]

    exp_dict = explainer.explain_instance(single_car)

    assert "base_market_value_inr" in exp_dict
    assert "final_predicted_price_inr" in exp_dict
    assert "feature_impacts" in exp_dict

    assert exp_dict["base_market_value_inr"] > 0
    assert exp_dict["final_predicted_price_inr"] > 0
    assert len(exp_dict["feature_impacts"]) > 0

    # Format text verification
    report_text = explainer.format_explanation_text(exp_dict)
    assert "SHAP EXPLANATION REPORT" in report_text
    assert "FINAL PREDICTED RESALE PRICE" in report_text
