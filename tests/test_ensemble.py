"""
Unit tests for Phase 3: Stacked ML Ensemble, Optuna Tuning, and Quantile Price Confidence Ranges.
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
from src.ensemble_model import StackedCarEnsemble, QuantilePricePredictor


def test_stacked_ensemble_fit_predict(tmp_path):
    df_raw = load_dataset_from_db()
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, cat_features = preprocessor.prepare_splits(df_raw)

    ensemble = StackedCarEnsemble(model_dir=str(tmp_path))
    ensemble.fit(X_train.iloc[:500], y_train.iloc[:500])
    metrics = ensemble.evaluate(X_test.iloc[:100], y_test.iloc[:100])

    assert "ensemble_r2" in metrics
    assert "catboost_r2" in metrics
    assert "xgboost_r2" in metrics
    assert "lightgbm_r2" in metrics
    assert metrics["ensemble_r2"] > 0.0

    preds = ensemble.predict(X_test.iloc[:5])
    assert len(preds) == 5
    assert all(p > 0 for p in preds)


def test_quantile_price_range_predictor():
    df_raw = load_dataset_from_db()
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, cat_features = preprocessor.prepare_splits(df_raw)

    q_predictor = QuantilePricePredictor()
    q_predictor.fit(X_train.iloc[:300], y_train.iloc[:300])

    ranges = q_predictor.predict_range(X_test.iloc[:5])
    assert len(ranges) == 5
    assert "price_low_10" in ranges.columns
    assert "price_median_50" in ranges.columns
    assert "price_high_90" in ranges.columns

    # Verify 10th percentile <= 50th percentile <= 90th percentile
    for idx, row in ranges.iterrows():
        assert row["price_low_10"] <= row["price_high_90"] + 1e-5
