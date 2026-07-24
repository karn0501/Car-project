"""
Phase 4 Execution Script: Explainable AI (SHAP) Valuation Breakdown Tool.
Loads dataset from database, runs SHAP TreeExplainer, and outputs human-readable
price contribution breakdowns for sample car listings.
"""

import os
import sys

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.data_exporter import load_dataset_from_db
from src.preprocessing import DataPreprocessor
from src.explainability import CarPriceExplainer


def run_explainability_demo():
    print("=" * 80)
    print("PHASE 4: EXPLAINABLE AI (SHAP) PRICE BREAKDOWN ENGINE")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1/3] Loading sample listings from database...")
    df_raw = load_dataset_from_db()
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, cat_features = preprocessor.prepare_splits(df_raw)

    # 2. Initialize Explainer
    print("\n[Step 2/3] Initializing SHAP TreeExplainer engine...")
    explainer = CarPriceExplainer()

    # 3. Explain 3 Sample Cars
    print("\n[Step 3/3] Generating SHAP Feature Breakdown Reports for 3 Cars:\n")
    sample_cars = X_test.iloc[:3]

    for idx in range(len(sample_cars)):
        single_car = sample_cars.iloc[[idx]]
        exp_dict = explainer.explain_instance(single_car)
        report_text = explainer.format_explanation_text(exp_dict)
        print(f"Sample Car #{idx+1} Explanation:")
        print(report_text)
        print("\n")


if __name__ == "__main__":
    run_explainability_demo()
