"""
Data Preprocessing & Feature Engineering Module for Used Car Price Prediction.
Engineers derived domain features (car age, price per KM, depreciation ratio, discontinued flag),
cleans statistical outliers, and splits dataset into reproducible train/test sets.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

CATEGORICAL_FEATURES = [
    "company_name",
    "model_name",
    "variant_name",
    "body_type",
    "fuel_type",
    "transmission",
    "city",
    "insurance_valid",
    "accident_history"
]

NUMERICAL_FEATURES = [
    "manufacture_year",
    "km_driven",
    "owner_count",
    "engine_cc",
    "seating_capacity",
    "ex_showroom_price",
    "car_age",
    "price_per_km",
    "depreciation_ratio",
    "is_discontinued"
]


class DataPreprocessor:
    """
    Handles feature engineering, outlier cleaning, and train/test dataset splitting.
    """

    def __init__(self, current_year: int = None):
        self.current_year = current_year if current_year else datetime.now().year

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates domain-specific engineered features.
        """
        df_proc = df.copy()

        # 1. Car Age in years
        df_proc["car_age"] = np.maximum(0, self.current_year - df_proc["manufacture_year"])

        # 2. Price per KM driven
        df_proc["price_per_km"] = df_proc["asking_price"] / np.maximum(1.0, df_proc["km_driven"])

        # 3. Ex-showroom Depreciation Ratio
        if "ex_showroom_price" in df_proc.columns:
            df_proc["depreciation_ratio"] = df_proc["asking_price"] / np.maximum(1.0, df_proc["ex_showroom_price"])
        else:
            df_proc["depreciation_ratio"] = 1.0

        # 4. Is Discontinued Model Flag
        if "model_discontinued_year" in df_proc.columns:
            df_proc["is_discontinued"] = df_proc["model_discontinued_year"].apply(lambda y: 1 if pd.notnull(y) else 0)
        else:
            df_proc["is_discontinued"] = 0

        # Fill missing categorical values with Unknown string
        for col in CATEGORICAL_FEATURES:
            if col in df_proc.columns:
                df_proc[col] = df_proc[col].fillna("Unknown").astype(str)

        # Fill missing numerical values with medians/defaults
        df_proc["engine_cc"] = df_proc["engine_cc"].fillna(1197)
        df_proc["seating_capacity"] = df_proc["seating_capacity"].fillna(5)
        df_proc["ex_showroom_price"] = df_proc["ex_showroom_price"].fillna(800000)

        return df_proc

    def clean_outliers(self, df: pd.DataFrame, lower_quantile=0.005, upper_quantile=0.995) -> pd.DataFrame:
        """
        Caps extreme price and mileage outliers using percentile thresholds.
        """
        df_clean = df.copy()

        if "asking_price" in df_clean.columns:
            p_low = df_clean["asking_price"].quantile(lower_quantile)
            p_high = df_clean["asking_price"].quantile(upper_quantile)
            df_clean = df_clean[(df_clean["asking_price"] >= p_low) & (df_clean["asking_price"] <= p_high)]

        if "km_driven" in df_clean.columns:
            km_high = df_clean["km_driven"].quantile(upper_quantile)
            df_clean["km_driven"] = np.minimum(df_clean["km_driven"], km_high)

        return df_clean

    def prepare_splits(self, df: pd.DataFrame, target_col: str = "asking_price", test_size: float = 0.2, random_state: int = 42):
        """
        Prepares X_train, X_test, y_train, y_test splits.
        """
        df_fe = self.engineer_features(df)
        df_fe = self.clean_outliers(df_fe)

        feature_cols = [c for c in (CATEGORICAL_FEATURES + NUMERICAL_FEATURES) if c in df_fe.columns]
        X = df_fe[feature_cols]
        y = df_fe[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test, CATEGORICAL_FEATURES
