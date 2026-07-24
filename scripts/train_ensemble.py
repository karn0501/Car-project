"""
Phase 3 Execution Script: Stacked ML Ensemble, Optuna Tuning & Quantile Confidence Intervals.
Loads dataset from database, tunes CatBoost/XGBoost/LightGBM hyper-parameters via Optuna,
trains Stacked Ensembled Meta-Model and Quantile Price Range Predictor, evaluates against
Phase 2 baseline, and persists all model artifacts to models/ directory.
"""

import os
import sys

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.data_exporter import load_dataset_from_db
from src.preprocessing import DataPreprocessor
from src.optuna_tuner import OptunaHyperparameterTuner
from src.ensemble_model import StackedCarEnsemble, QuantilePricePredictor


def run_ensemble_pipeline():
    print("=" * 85)
    print("PHASE 3: STACKED ML ENSEMBLE, OPTUNA TUNING & QUANTILE PRICE RANGE PIPELINE")
    print("=" * 85)

    # 1. Load Data
    print("\n[Step 1/5] Loading structured resale records from database...")
    df_raw = load_dataset_from_db()
    print(f"Loaded {len(df_raw)} records across {len(df_raw['company_name'].unique())} brands.")

    # 2. Preprocess & Split
    print("\n[Step 2/5] Engineering features and preparing 80/20 train/test split...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, cat_features = preprocessor.prepare_splits(df_raw)
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Fast Optuna Tuning on Representative Sample
    print("\n[Step 3/5] Running Optuna automated hyperparameter optimization...")
    tuner = OptunaHyperparameterTuner(n_trials=2, random_state=42)
    sample_size = min(4000, len(X_train))
    X_tune, y_tune = X_train.iloc[:sample_size], y_train.iloc[:sample_size]

    best_cb = tuner.tune_catboost(X_tune, y_tune, cat_features=cat_features)
    best_xgb = tuner.tune_xgboost(X_tune, y_tune, cat_features=cat_features)
    best_lgb = tuner.tune_lightgbm(X_tune, y_tune, cat_features=cat_features)
    tuner.save_best_params()
    print(f"Optuna Hyperparameter Tuning Complete! Best params saved to models/optuna_best_params.json")

    # 4. Train Stacked Ensemble on Full Training Set
    print("\n[Step 4/5] Training Stacked Ensemble (CatBoost + XGBoost + LightGBM -> Linear Meta-Model)...")
    ensemble = StackedCarEnsemble(model_dir="models")
    ensemble.fit(X_train, y_train, best_params=tuner.best_params)
    metrics = ensemble.evaluate(X_test, y_test)
    saved_ensemble_path = ensemble.save_ensemble()

    # 5. Train Quantile Range Predictor
    print("\n[Step 5/5] Training Quantile Regressors for Price Ranges (10th, 50th, 90th percentiles)...")
    q_predictor = QuantilePricePredictor()
    q_predictor.fit(X_train, y_train)
    q_ranges = q_predictor.predict_range(X_test.iloc[:5])

    print("\n" + "=" * 85)
    print("PHASE 3 ENSEMBLE PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 85)
    print("Individual Model Performance (R² Score):")
    print(f"  |-- CatBoost Regressor  : {metrics['catboost_r2'] * 100:.2f}% (R² = {metrics['catboost_r2']})")
    print(f"  |-- XGBoost Regressor   : {metrics['xgboost_r2'] * 100:.2f}% (R² = {metrics['xgboost_r2']})")
    print(f"  \\-- LightGBM Regressor  : {metrics['lightgbm_r2'] * 100:.2f}% (R² = {metrics['lightgbm_r2']})")
    print("-" * 85)
    print("STACKED META-MODEL FINAL ACCURACY:")
    print(f"  |-- Ensemble RMSE       : INR {metrics['ensemble_rmse']:,.2f}")
    print(f"  |-- Ensemble MAE        : INR {metrics['ensemble_mae']:,.2f}")
    print(f"  |-- Ensemble R² Accuracy: {metrics['ensemble_r2'] * 100:.2f}% (R² = {metrics['ensemble_r2']})")
    print(f"  \\-- Meta-Model Weights  : CatBoost: {metrics['meta_model_weights'][0]:.3f}, XGBoost: {metrics['meta_model_weights'][1]:.3f}, LightGBM: {metrics['meta_model_weights'][2]:.3f}")
    print("-" * 85)
    print("Sample Quantile Price Confidence Ranges for 5 Resale Cars:")
    for idx, row in q_ranges.iterrows():
        print(f"  Car #{idx+1}: Low (10th%): INR {row['price_low_10']:,.0f} | Median (50th%): INR {row['price_median_50']:,.0f} | High (90th%): INR {row['price_high_90']:,.0f}")
    print("=" * 85)
    print(f"Saved Ensemble Model    : {saved_ensemble_path}")
    print("=" * 85)


if __name__ == "__main__":
    run_ensemble_pipeline()
