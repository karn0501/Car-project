"""
Optuna Automated Hyperparameter Tuning Module for Used Car Price Prediction.
Performs Optuna study optimizations for CatBoost, XGBoost, and LightGBM base models.
"""

import os
import json
import optuna
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaHyperparameterTuner:
    """
    Automated hyperparameter optimizer using Optuna.
    """

    def __init__(self, n_trials: int = 3, random_state: int = 42):
        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params = {}
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def tune_catboost(self, X: pd.DataFrame, y: pd.Series, cat_features: list) -> dict:
        """
        Tunes CatBoost hyper-parameters.
        """
        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 150, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.15),
                "depth": trial.suggest_int("depth", 4, 7),
                "random_seed": self.random_state,
                "verbose": 0
            }

            kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            rmse_scores = []

            for train_idx, val_idx in kf.split(X):
                X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                cats = [c for c in cat_features if c in X_tr.columns]
                for col in cats:
                    X_tr[col] = X_tr[col].astype(str)
                    X_val[col] = X_val[col].astype(str)

                model = CatBoostRegressor(**params)
                model.fit(X_tr, y_tr, cat_features=cats)
                preds = model.predict(X_val)
                rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

            return float(np.mean(rmse_scores))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        self.best_params["catboost"] = study.best_params
        return study.best_params

    def tune_xgboost(self, X: pd.DataFrame, y: pd.Series, cat_features: list) -> dict:
        """
        Tunes XGBoost hyper-parameters.
        """
        X_enc = X.copy()
        cats = [c for c in cat_features if c in X_enc.columns]
        X_enc[cats] = self.encoder.fit_transform(X_enc[cats].astype(str))

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 250),
                "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.15),
                "max_depth": trial.suggest_int("max_depth", 4, 7),
                "random_state": self.random_state,
                "n_jobs": -1
            }

            kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            rmse_scores = []

            for train_idx, val_idx in kf.split(X_enc):
                X_tr, X_val = X_enc.iloc[train_idx], X_enc.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

            return float(np.mean(rmse_scores))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        self.best_params["xgboost"] = study.best_params
        return study.best_params

    def tune_lightgbm(self, X: pd.DataFrame, y: pd.Series, cat_features: list) -> dict:
        """
        Tunes LightGBM hyper-parameters.
        """
        X_enc = X.copy()
        cats = [c for c in cat_features if c in X_enc.columns]
        X_enc[cats] = self.encoder.fit_transform(X_enc[cats].astype(str))

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 250),
                "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.15),
                "num_leaves": trial.suggest_int("num_leaves", 20, 60),
                "max_depth": trial.suggest_int("max_depth", 4, 7),
                "random_state": self.random_state,
                "verbose": -1
            }

            kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            rmse_scores = []

            for train_idx, val_idx in kf.split(X_enc):
                X_tr, X_val = X_enc.iloc[train_idx], X_enc.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

            return float(np.mean(rmse_scores))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        self.best_params["lightgbm"] = study.best_params
        return study.best_params

    def save_best_params(self, filepath: str = "models/optuna_best_params.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.best_params, f, indent=4)
