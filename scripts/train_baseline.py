"""
Phase 2 Baseline Model Training Execution Script.
Loads dataset from database, preprocesses & engineers domain features,
trains CatBoost baseline regressor, logs evaluation metrics (RMSE, MAE, R²),
and saves trained model artifact to models/ directory.
"""

import os
import sys

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.data_exporter import load_dataset_from_db
from src.preprocessing import DataPreprocessor
from src.baseline_model import BaselineCatBoostModel


def run_baseline_pipeline():
    print("=" * 80)
    print("PHASE 2: BASELINE ML MODEL TRAINING PIPELINE")
    print("=" * 80)

    # 1. Load Data from DB
    print("\n[Step 1/4] Loading structured resale records from database...")
    df_raw = load_dataset_from_db()
    print(f"Loaded {len(df_raw)} records across {len(df_raw['company_name'].unique())} brands.")

    # 2. Preprocess & Engineer Features
    print("\n[Step 2/4] Preprocessing data and engineering features (car_age, price_per_km, depreciation_ratio)...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, cat_features = preprocessor.prepare_splits(df_raw)
    print(f"Training split size: {len(X_train)} samples | Test split size: {len(X_test)} samples")

    # 3. Train Baseline Model
    print("\n[Step 3/4] Training Baseline CatBoost Regressor...")
    model_wrapper = BaselineCatBoostModel(model_dir="models")
    model_wrapper.fit(X_train, y_train, cat_features=cat_features)

    # 4. Evaluate Metrics
    print("\n[Step 4/4] Evaluating baseline metrics on test dataset...")
    metrics = model_wrapper.evaluate(X_test, y_test)

    saved_path = model_wrapper.save_model()

    print("\n" + "=" * 80)
    print("PHASE 2 BASELINE TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Model Architecture  : {metrics['model_type']}")
    print(f"Test Set Size       : {metrics['test_samples']} listings")
    print(f"Root Mean Sq Error  : INR {metrics['rmse']:,.2f} (RMSE)")
    print(f"Mean Absolute Error : INR {metrics['mae']:,.2f} (MAE)")
    print(f"R2 Accuracy Score   : {metrics['r2_score'] * 100:.2f}% (R² = {metrics['r2_score']})")
    print(f"Saved Model Binary  : {saved_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_baseline_pipeline()
