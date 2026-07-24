"""
Explainable AI (XAI) Module for Used Car Price Prediction (Phase 4).
Calculates exact SHAP feature contribution values using CatBoost's native C++ engine
and formats predictions into human-readable price breakdown reports.
"""

import os
import joblib
import numpy as np
import pandas as pd
from catboost import Pool

from src.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


class CarPriceExplainer:
    """
    SHAP & Feature Impact Explainability Engine for interpreting car price predictions.
    Uses CatBoost's native C++ SHAP calculation engine for zero DLL/dependency overhead.
    """

    def __init__(self, ensemble_model_path: str = "models/ensemble_stack.joblib"):
        self.ensemble_path = os.path.abspath(ensemble_model_path)
        if not os.path.exists(self.ensemble_path):
            raise FileNotFoundError(f"Ensemble model not found at {self.ensemble_path}. Please train Phase 3 first.")

        self.ensemble = joblib.load(self.ensemble_path)
        self.cat_features = CATEGORICAL_FEATURES

    def explain_instance(self, car_features_df: pd.DataFrame) -> dict:
        """
        Computes SHAP values and returns a human-readable price contribution breakdown.
        Args:
            car_features_df (pd.DataFrame): Single row DataFrame with car attributes.
        Returns:
            dict: Structured breakdown with base_value, feature_impacts, and final_prediction.
        """
        df_cb = car_features_df.copy()
        cats = [c for c in self.cat_features if c in df_cb.columns]
        for col in cats:
            df_cb[col] = df_cb[col].astype(str)

        cb_pool = Pool(df_cb, cat_features=cats)

        # Native CatBoost C++ SHAP values matrix calculation
        shap_vals_matrix = self.ensemble.catboost_model.get_feature_importance(cb_pool, type="ShapValues")

        # The last column in CatBoost ShapValues matrix is the baseline expected value
        base_value = float(shap_vals_matrix[0, -1])
        feature_shap_vals = shap_vals_matrix[0, :-1]

        predicted_price = float(self.ensemble.predict(car_features_df)[0])

        feature_impacts = {}
        for col_name, val, shap_val in zip(df_cb.columns, df_cb.iloc[0].values, feature_shap_vals):
            feature_impacts[col_name] = {
                "feature_value": str(val),
                "shap_impact_inr": float(shap_val)
            }

        # Sort impacts by absolute magnitude
        sorted_impacts = dict(
            sorted(feature_impacts.items(), key=lambda x: abs(x[1]["shap_impact_inr"]), reverse=True)
        )

        return {
            "base_market_value_inr": round(base_value, 2),
            "final_predicted_price_inr": round(predicted_price, 2),
            "feature_impacts": sorted_impacts
        }

    def format_explanation_text(self, explanation_dict: dict) -> str:
        """
        Formats explanation dictionary into a clean, human-readable text output.
        """
        base_val = explanation_dict["base_market_value_inr"]
        final_price = explanation_dict["final_predicted_price_inr"]
        impacts = explanation_dict["feature_impacts"]

        lines = []
        lines.append("=" * 75)
        lines.append("        USED CAR PRICE PREDICTION - SHAP EXPLANATION REPORT")
        lines.append("=" * 75)
        lines.append(f"Base Average Market Price : INR {base_val:,.2f}")
        lines.append("-" * 75)
        lines.append("Individual Feature Price Contributions:")

        for feat, info in impacts.items():
            val = info["feature_value"]
            imp = info["shap_impact_inr"]
            sign = "+" if imp >= 0 else "-"
            lines.append(f"  • {feat:<22} ({val:<15}): {sign} INR {abs(imp):,.2f}")

        lines.append("-" * 75)
        lines.append(f"FINAL PREDICTED RESALE PRICE  : INR {final_price:,.2f}")
        lines.append("=" * 75)

        return "\n".join(lines)
