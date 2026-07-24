"""
Stacked ML Ensemble & Quantile Range Predictor Module (Phase 3).
Combines CatBoost, XGBoost, and LightGBM base models with Ordinal Encoding
for ultra-fast performance, 5-fold out-of-fold cross validation, and Linear Meta-Model.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb

from src.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


class StackedCarEnsemble:
    """
    Stacked Ensemble Model (CatBoost + XGBoost + LightGBM -> Linear Meta-Model).
    """

    def __init__(self, model_dir: str = "models", random_state: int = 42):
        self.model_dir = os.path.abspath(model_dir)
        os.makedirs(self.model_dir, exist_ok=True)
        self.random_state = random_state

        self.catboost_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.meta_model = Ridge(alpha=1.0)
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        self.cat_features = CATEGORICAL_FEATURES
        self.metrics = {}

    def _prepare_encoded(self, df: pd.DataFrame, fit_encoder: bool = False) -> pd.DataFrame:
        df_enc = df.copy()
        cats = [c for c in self.cat_features if c in df_enc.columns]

        if fit_encoder:
            df_enc[cats] = self.encoder.fit_transform(df_enc[cats].astype(str))
        else:
            df_enc[cats] = self.encoder.transform(df_enc[cats].astype(str))

        return df_enc

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, best_params: dict = None):
        """
        Trains CatBoost, XGBoost, and LightGBM base models and fits the linear meta-model.
        """
        # 1. Prepare Categorical & Ordinal Encoded Data
        X_train_cb = X_train.copy()
        cats = [c for c in self.cat_features if c in X_train_cb.columns]
        for col in cats:
            X_train_cb[col] = X_train_cb[col].astype(str)

        X_train_enc = self._prepare_encoded(X_train, fit_encoder=True)

        # 2. Extract Hyperparameters
        cb_params = best_params.get("catboost", {}) if best_params else {}
        xgb_params = best_params.get("xgboost", {}) if best_params else {}
        lgb_params = best_params.get("lightgbm", {}) if best_params else {}

        cb_defaults = {"iterations": 120, "learning_rate": 0.1, "depth": 5, "random_seed": self.random_state, "verbose": 0}
        xgb_defaults = {"n_estimators": 120, "learning_rate": 0.1, "max_depth": 5, "random_state": self.random_state, "n_jobs": -1}
        lgb_defaults = {"n_estimators": 120, "learning_rate": 0.1, "num_leaves": 31, "max_depth": 5, "random_state": self.random_state, "verbose": -1}

        cb_defaults.update(cb_params)
        xgb_defaults.update(xgb_params)
        lgb_defaults.update(lgb_params)

        # 3. 3-Fold Out-Of-Fold Predictions Generation
        kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        oof_cb = np.zeros(len(X_train))
        oof_xgb = np.zeros(len(X_train))
        oof_lgb = np.zeros(len(X_train))

        for train_idx, val_idx in kf.split(X_train):
            # CatBoost Fold
            X_tr_cb, X_val_cb = X_train_cb.iloc[train_idx], X_train_cb.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            m_cb = CatBoostRegressor(**cb_defaults)
            m_cb.fit(X_tr_cb, y_tr, cat_features=cats)
            oof_cb[val_idx] = m_cb.predict(X_val_cb)

            # XGBoost & LightGBM Folds
            X_tr_enc, X_val_enc = X_train_enc.iloc[train_idx], X_train_enc.iloc[val_idx]

            m_xgb = xgb.XGBRegressor(**xgb_defaults)
            m_xgb.fit(X_tr_enc, y_tr)
            oof_xgb[val_idx] = m_xgb.predict(X_val_enc)

            m_lgb = lgb.LGBMRegressor(**lgb_defaults)
            m_lgb.fit(X_tr_enc, y_tr)
            oof_lgb[val_idx] = m_lgb.predict(X_val_enc)

        # 4. Meta-Model Training on OOF Matrix
        OOF_matrix = np.column_stack((oof_cb, oof_xgb, oof_lgb))
        self.meta_model.fit(OOF_matrix, y_train)

        # 5. Full Retrain Base Models
        self.catboost_model = CatBoostRegressor(**cb_defaults)
        self.catboost_model.fit(X_train_cb, y_train, cat_features=cats)

        self.xgb_model = xgb.XGBRegressor(**xgb_defaults)
        self.xgb_model.fit(X_train_enc, y_train)

        self.lgb_model = lgb.LGBMRegressor(**lgb_defaults)
        self.lgb_model.fit(X_train_enc, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predicts asking prices via stacked base models -> linear meta-model.
        """
        X_test_cb = X_test.copy()
        cats = [c for c in self.cat_features if c in X_test_cb.columns]
        for col in cats:
            X_test_cb[col] = X_test_cb[col].astype(str)

        X_test_enc = self._prepare_encoded(X_test, fit_encoder=False)

        p_cb = self.catboost_model.predict(X_test_cb)
        p_xgb = self.xgb_model.predict(X_test_enc)
        p_lgb = self.lgb_model.predict(X_test_enc)

        base_preds_matrix = np.column_stack((p_cb, p_xgb, p_lgb))
        final_preds = self.meta_model.predict(base_preds_matrix)

        return final_preds

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates individual base models vs stacked ensemble accuracy.
        """
        X_test_cb = X_test.copy()
        cats = [c for c in self.cat_features if c in X_test_cb.columns]
        for col in cats:
            X_test_cb[col] = X_test_cb[col].astype(str)

        X_test_enc = self._prepare_encoded(X_test, fit_encoder=False)

        p_cb = self.catboost_model.predict(X_test_cb)
        p_xgb = self.xgb_model.predict(X_test_enc)
        p_lgb = self.lgb_model.predict(X_test_enc)

        final_preds = self.predict(X_test)

        self.metrics = {
            "catboost_r2": round(float(r2_score(y_test, p_cb)), 4),
            "xgboost_r2": round(float(r2_score(y_test, p_xgb)), 4),
            "lightgbm_r2": round(float(r2_score(y_test, p_lgb)), 4),
            "ensemble_rmse": round(float(np.sqrt(mean_squared_error(y_test, final_preds))), 2),
            "ensemble_mae": round(float(mean_absolute_error(y_test, final_preds)), 2),
            "ensemble_r2": round(float(r2_score(y_test, final_preds)), 4),
            "test_samples": len(y_test),
            "meta_model_weights": [round(float(w), 4) for w in self.meta_model.coef_]
        }

        return self.metrics

    def save_ensemble(self, filename: str = "ensemble_stack.joblib"):
        save_path = os.path.join(self.model_dir, filename)
        joblib.dump(self, save_path)

        metrics_path = os.path.join(self.model_dir, "ensemble_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=4)

        return save_path


class QuantilePricePredictor:
    """
    Quantile Regression Price Predictor for 10th, 50th, and 90th percentile ranges.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model_low = CatBoostRegressor(loss_function="Quantile:alpha=0.10", iterations=100, random_seed=random_state, verbose=0)
        self.model_med = CatBoostRegressor(loss_function="Quantile:alpha=0.50", iterations=100, random_seed=random_state, verbose=0)
        self.model_high = CatBoostRegressor(loss_function="Quantile:alpha=0.90", iterations=100, random_seed=random_state, verbose=0)
        self.cat_features = CATEGORICAL_FEATURES

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        X_tr = X_train.copy()
        cats = [c for c in self.cat_features if c in X_tr.columns]
        for col in cats:
            X_tr[col] = X_tr[col].astype(str)

        self.model_low.fit(X_tr, y_train, cat_features=cats)
        self.model_med.fit(X_tr, y_train, cat_features=cats)
        self.model_high.fit(X_tr, y_train, cat_features=cats)

    def predict_range(self, X_test: pd.DataFrame) -> pd.DataFrame:
        X_t = X_test.copy()
        cats = [c for c in self.cat_features if c in X_t.columns]
        for col in cats:
            X_t[col] = X_t[col].astype(str)

        p_low = np.maximum(40000.0, self.model_low.predict(X_t))
        p_med = np.maximum(40000.0, self.model_med.predict(X_t))
        p_high = np.maximum(40000.0, self.model_high.predict(X_t))

        return pd.DataFrame({
            "price_low_10": p_low,
            "price_median_50": p_med,
            "price_high_90": p_high
        })
