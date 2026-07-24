"""
Phase 11: Regional Tax & RTO Fee Calculator.
Calculates state-specific RTO road tax rates, interstate transfer NOC costs,
and Delhi NCR 10-year diesel rule depreciation penalties.
"""

from typing import Dict, Any


# State RTO Tax percentage estimates on resale transfer
STATE_RTO_TAX_RATES = {
    "Delhi": 0.08,
    "Mumbai": 0.12,
    "Pune": 0.12,
    "Bangalore": 0.15,
    "Hyderabad": 0.12,
    "Chennai": 0.11,
    "Ahmedabad": 0.09,
    "Kolkata": 0.10,
    "Jaipur": 0.09,
    "Lucknow": 0.08
}


class RegionalTaxCalculator:
    """
    State RTO Road Tax & Regulatory Fee Engine.
    """

    def calculate_regional_fees(self, city: str, base_price: float,
                                 fuel_type: str = "Petrol", car_age: int = 4) -> Dict[str, Any]:
        """
        Calculates estimated RTO road tax, transfer fees, and environmental surcharges.
        """
        city_cap = city.capitalize() if city else "Mumbai"
        tax_rate = STATE_RTO_TAX_RATES.get(city_cap, 0.10)

        # Base RTO Transfer Tax
        rto_tax = base_price * tax_rate

        # Delhi NCR 10-year Diesel Rule Depreciation Penalty
        ncr_penalty = 0.0
        is_ncr = city_cap in ["Delhi", "Gurgaon", "Noida", "Faridabad"]
        if is_ncr and fuel_type == "Diesel" and car_age >= 8:
            # 25% value drop as vehicle approaches 10-year ban threshold
            ncr_penalty = base_price * 0.25

        # Electric Vehicle RTO Subsidy / Waiver (0% Tax)
        if fuel_type == "Electric":
            rto_tax = 0.0

        # State NOC & Smartcard Transfer Admin Fee
        admin_fee = 3500.0

        total_regional_cost = rto_tax + ncr_penalty + admin_fee
        final_buyer_cost = base_price + total_regional_cost

        return {
            "city": city_cap,
            "base_vehicle_price": round(base_price, 0),
            "rto_tax_rate_pct": round(tax_rate * 100, 1),
            "estimated_rto_tax": round(rto_tax, 0),
            "ncr_diesel_penalty": round(ncr_penalty, 0),
            "transfer_admin_fee": round(admin_fee, 0),
            "total_regional_cost": round(total_regional_cost, 0),
            "total_buyer_landed_cost": round(final_buyer_cost, 0)
        }
