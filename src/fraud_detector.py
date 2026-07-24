"""
Phase 12: Real-Time Fraud & Anomaly Detector.
Detects odometer rollbacks/tampering, extreme price manipulation, and suspicious listing patterns.
Returns Fraud Risk Score (0.00–1.00), Risk Level (LOW, MODERATE, HIGH), and detected anomaly flags.
"""

from typing import Dict, Any, List
from datetime import datetime


class ListingFraudDetector:
    """
    Used Car Listing Fraud & Odometer Tampering Anomaly Detector.
    """

    def evaluate_listing_fraud(self, car_data: Dict[str, Any], predicted_price: float = None) -> Dict[str, Any]:
        """
        Evaluates a vehicle listing for fraud risk and odometer tampering.
        """
        flags: List[str] = []
        risk_score = 0.0

        current_year = datetime.now().year
        year = int(car_data.get("manufacture_year", 2020))
        km = float(car_data.get("km_driven", 35000))
        age = max(1, current_year - year)
        asking_price = float(car_data.get("asking_price", 0))

        # 1. Odometer Tampering Checks
        km_per_year = km / age

        if age >= 5 and km < 5000:
            flags.append("SUSPICIOUS_LOW_ODOMETER: Unusually low mileage for vehicle age (< 1,000 km/year)")
            risk_score += 0.35
        elif age >= 3 and km_per_year < 1500:
            flags.append("POTENTIAL_ODOMETER_ROLLBACK: Mileage below normal annual average (< 1,500 km/year)")
            risk_score += 0.20

        if km > 450000:
            flags.append("EXTREME_HIGH_MILEAGE: Vehicle has exceeded typical mechanical lifespan threshold")
            risk_score += 0.15

        # 2. Price Manipulation Checks
        if predicted_price and asking_price > 0:
            ratio = asking_price / predicted_price

            if ratio < 0.50:
                flags.append("TOO_GOOD_TO_BE_TRUE: Price is > 50% below fair market value (Potential Scam/Stolen)")
                risk_score += 0.45
            elif ratio < 0.70:
                flags.append("UNDERPRICED_ANOMALY: Price is significantly below market average (> 30% discount)")
                risk_score += 0.20
            elif ratio > 1.60:
                flags.append("OVERPRICED_ANOMALY: Price is > 60% above fair market valuation")
                risk_score += 0.15

        # 3. Ownership / Document Checks
        owners = int(car_data.get("owner_count", 1))
        if age <= 2 and owners >= 3:
            flags.append("FREQUENT_OWNERSHIP_TRANSFER: Vehicle transferred 3+ times in less than 2 years")
            risk_score += 0.25

        accident = str(car_data.get("accident_history", "No")).capitalize()
        if accident == "Yes":
            flags.append("ACCIDENT_HISTORY_REPORTED: Vehicle has recorded accident/structural damage")
            risk_score += 0.15

        # Final Score Cap
        risk_score = round(min(1.0, max(0.0, risk_score)), 2)

        # Risk Classification
        if risk_score >= 0.50:
            risk_level = "HIGH"
        elif risk_score >= 0.25:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return {
            "fraud_risk_score": risk_score,
            "risk_level": risk_level,
            "anomaly_flags": flags,
            "anomaly_count": len(flags),
            "km_per_year": round(km_per_year, 1),
            "vehicle_age": age,
            "is_safe_listing": risk_level != "HIGH"
        }
