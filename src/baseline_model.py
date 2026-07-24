"""
Baseline ML Model Module for Used Car Price Prediction (Phase 2).
Implements BaselineCatBoostModel with native categorical feature handling,
metrics evaluation (RMSE, MAE, R²), and model artifact persistence.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    from sklearn.ensemble import GradientBoostingRegressor

from src.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


class BaselineCatBoostModel:
    """
    CatBoost Regressor Baseline Model wrapper for car price prediction.
    """

    def __init__(self, model_dir: str = "models", random_state: int = 42):
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.random_state = random_state

        if HAS_CATBOOST:
            self.model = CatBoostRegressor(
                iterations=500,
                learning_rate=0.08,
                depth=6,
                random_seed=self.random_state,
                verbose=100
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.08,
                max_depth=5,
                random_state=self.random_state
            )

        self.feature_names = []
        self.cat_features = []
        self.metrics = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, cat_features: list = None):
        """
        Fits the baseline model on training features and target.
        """
        self.feature_names = list(X_train.columns)
        self.cat_features = [c for c in (cat_features if cat_features else CATEGORICAL_FEATURES) if c in X_train.columns]

        if HAS_CATBOOST:
            # Ensure categorical columns are strings for CatBoost
            X_train_cb = X_train.copy()
            for col in self.cat_features:
                X_train_cb[col] = X_train_cb[col].astype(str)

            self.model.fit(X_train_cb, y_train, cat_features=self.cat_features)
        else:
            # Fallback one-hot encoding for GradientBoosting
            X_train_encoded = pd.get_dummies(X_train, columns=self.cat_features, drop_first=True)
            self.feature_names = list(X_train_encoded.columns)
            self.model.fit(X_train_encoded, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates model accuracy on test dataset using RMSE, MAE, and R².
        """
        if HAS_CATBOOST:
            X_test_cb = X_test.copy()
            for col in self.cat_features:
                X_test_cb[col] = X_test_cb[col].astype(str)
            y_pred = self.model.predict(X_test_cb)
        else:
            X_test_encoded = pd.get_dummies(X_test, columns=self.cat_features, drop_first=True)
            X_test_encoded = X_test_encoded.reindex(columns=self.feature_names, fill_value=0)
            y_pred = self.model.predict(X_test_encoded)

        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        self.metrics = {
            "model_type": "CatBoostRegressor" if HAS_CATBOOST else "GradientBoostingRegressor",
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2_score": round(r2, 4),
            "test_samples": len(y_test),
            "trained_at": datetime.now().isoformat()
        }

        return self.metrics

    def predict(self, df_input: pd.DataFrame) -> np.ndarray:
        """
        Predicts asking prices for input DataFrame features.
        """
        if HAS_CATBOOST:
            X_in = df_input.copy()
            for col in self.cat_features:
                if col in X_in.columns:
                    X_in[col] = X_in[col].astype(str)
            return self.model.predict(X_in)
        else:
            X_in_encoded = pd.get_dummies(df_input, columns=self.cat_features, drop_first=True)
            X_in_encoded = X_in_encoded.reindex(columns=self.feature_names, fill_value=0)
            return self.model.predict(X_in_encoded)

    def save_model(self, filename: str = "baseline_catboost.cbm") -> str:
        """
        Saves the trained model binary and metadata file.
        """
        model_path = os.path.join(self.model_dir, filename)
        metrics_path = os.path.join(self.model_dir, "baseline_metrics.json")

        if HAS_CATBOOST:
            self.model.save_model(model_path)
        else:
            joblib.dump(self.model, os.path.join(self.model_dir, "baseline_model.joblib"))

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=4)

        return model_path
