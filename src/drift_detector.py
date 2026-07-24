"""
Phase 10: Model Data Drift Detector Engine.
Monitors input feature distributions (e.g. km_driven, manufacture_year, engine_cc, price)
using statistical Kolmogorov-Smirnov (KS-test) and Population Stability Index (PSI).
Triggers retraining alerts when live prediction data drifts significantly from training baseline.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.stats import ks_2samp


DRIFT_REPORT_FILE = os.path.abspath("models/drift_report.json")


class ModelDriftDetector:
    """
    Statistical Data Drift Engine for ML Model Health & Monitoring.
    """

    def __init__(self, baseline_df: pd.DataFrame = None):
        self.baseline_df = baseline_df
        if self.baseline_df is None:
            self._load_baseline_from_db()

    def _load_baseline_from_db(self):
        """Loads baseline dataset from local database."""
        try:
            from db.data_exporter import load_dataset_from_db
            self.baseline_df = load_dataset_from_db()
        except Exception:
            # Synthetic baseline fallback for standalone execution
            self.baseline_df = pd.DataFrame({
                "manufacture_year": np.random.randint(2012, 2024, 1000),
                "km_driven": np.random.normal(45000, 20000, 1000).clip(1000, 300000),
                "engine_cc": np.random.choice([1197, 1497, 1995, 1493], 1000),
                "asking_price": np.random.normal(650000, 300000, 1000).clip(100000, 3000000),
            })

    def calculate_psi(self, baseline: np.ndarray, target: np.ndarray, num_bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) between baseline and target sample.
        PSI < 0.1  : No significant distribution change
        PSI 0.1-0.2: Moderate drift
        PSI > 0.2  : Significant drift (Retraining Required)
        """
        if len(baseline) == 0 or len(target) == 0:
            return 0.0

        b_min = min(baseline.min(), target.min())
        b_max = max(baseline.max(), target.max())
        if b_min == b_max:
            return 0.0

        bins = np.linspace(b_min, b_max, num_bins + 1)

        b_counts, _ = np.histogram(baseline, bins=bins)
        t_counts, _ = np.histogram(target, bins=bins)

        # Convert to percentages with zero-smoothing
        b_pct = (b_counts + 1e-4) / (len(baseline) + 1e-4 * num_bins)
        t_pct = (t_counts + 1e-4) / (len(target) + 1e-4 * num_bins)

        psi = np.sum((t_pct - b_pct) * np.log(t_pct / b_pct))
        return round(float(psi), 4)

    def detect_feature_drift(self, target_df: pd.DataFrame, feature_name: str) -> Dict[str, Any]:
        """
        Analyze statistical drift for a specific numerical feature.
        """
        if feature_name not in self.baseline_df.columns or feature_name not in target_df.columns:
            return {"feature": feature_name, "status": "SKIPPED", "reason": "Feature not present in dataset"}

        base_vals = self.baseline_df[feature_name].dropna().values.astype(float)
        targ_vals = target_df[feature_name].dropna().values.astype(float)

        if len(base_vals) < 5 or len(targ_vals) < 5:
            return {"feature": feature_name, "status": "INSUFFICIENT_DATA"}

        # 1. KS-test (p-value < 0.05 indicates different distributions)
        ks_stat, p_value = ks_2samp(base_vals, targ_vals)

        # 2. Population Stability Index (PSI)
        psi_score = self.calculate_psi(base_vals, targ_vals)

        is_drifted = bool((p_value < 0.05) and (psi_score > 0.1))

        return {
            "feature": feature_name,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "psi_score": psi_score,
            "is_drifted": is_drifted,
            "drift_severity": "HIGH" if psi_score > 0.2 else ("MODERATE" if psi_score > 0.1 else "LOW")
        }

    def evaluate_dataset_health(self, live_data_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs comprehensive data drift analysis across core numerical features.
        """
        features_to_check = ["manufacture_year", "km_driven", "engine_cc", "asking_price"]
        results = []
        drifted_count = 0

        for feat in features_to_check:
            if feat in live_data_df.columns and feat in self.baseline_df.columns:
                res = self.detect_feature_drift(live_data_df, feat)
                results.append(res)
                if res.get("is_drifted"):
                    drifted_count += 1

        overall_status = "DRIFT_DETECTED" if drifted_count > 0 else "HEALTHY"
        recommendation = "Trigger automated model retraining" if drifted_count > 0 else "Model predictions remain calibrated"

        report = {
            "overall_status": overall_status,
            "drifted_feature_count": drifted_count,
            "checked_feature_count": len(results),
            "recommendation": recommendation,
            "feature_details": results,
            "timestamp": pd.Timestamp.now().isoformat()
        }

        # Save to report file
        try:
            os.makedirs(os.path.dirname(DRIFT_REPORT_FILE), exist_ok=True)
            with open(DRIFT_REPORT_FILE, "w") as f:
                json.dump(report, f, indent=2)
        except Exception:
            pass

        return report
