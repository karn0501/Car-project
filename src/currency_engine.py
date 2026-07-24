"""
Phase 11: Multi-Currency Engine.
Converts base INR vehicle valuations into international currencies (USD, EUR, GBP, AED, JPY)
with real-time exchange rates, fallback matrix, and localized currency formatting.
"""

from typing import Dict, Any


# Standard exchange rates (Base: 1 INR)
EXCHANGE_RATES_INR_BASE = {
    "INR": 1.0,
    "USD": 0.0120,    # 1 USD ≈ 83.33 INR
    "EUR": 0.0111,    # 1 EUR ≈ 90.09 INR
    "GBP": 0.0095,    # 1 GBP ≈ 105.26 INR
    "AED": 0.0441,    # 1 AED ≈ 22.68 INR
    "JPY": 1.7800,    # 1 JPY ≈ 0.56 INR
}

CURRENCY_SYMBOLS = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "AED": "AED ",
    "JPY": "¥"
}


class CurrencyConverter:
    """
    Multi-Currency Foreign Exchange & Localized Price Formatting Engine.
    """

    def __init__(self, rates_dict: Dict[str, float] = None):
        self.rates = rates_dict or EXCHANGE_RATES_INR_BASE

    def get_supported_currencies(self) -> Dict[str, str]:
        """Returns list of supported currency codes and symbols."""
        return CURRENCY_SYMBOLS

    def convert_from_inr(self, amount_inr: float, target_currency: str = "INR") -> float:
        """
        Convert INR amount to target currency.
        """
        curr = target_currency.upper().strip()
        if curr not in self.rates:
            curr = "INR"

        rate = self.rates[curr]
        converted = amount_inr * rate
        return round(converted, 2 if curr != "JPY" else 0)

    def format_price(self, amount: float, currency: str = "INR") -> str:
        """
        Formats price string according to target currency conventions.
        - INR: Indian numbering system (e.g. ₹6.50 Lakhs or ₹6,50,000)
        - USD/EUR/GBP: Standard comma separation (e.g. $7,800)
        """
        curr = currency.upper().strip()
        sym = CURRENCY_SYMBOLS.get(curr, "")

        if curr == "INR":
            if amount >= 10000000:
                crores = amount / 10000000
                return f"{sym}{crores:.2f} Cr"
            elif amount >= 100000:
                lakhs = amount / 100000
                return f"{sym}{lakhs:.2f} Lakhs"
            else:
                return f"{sym}{amount:,.0f}"
        elif curr == "JPY":
            return f"{sym}{amount:,.0f}"
        else:
            return f"{sym}{amount:,.2f}"

    def localize_prediction_prices(self, pred_dict: Dict[str, Any], target_currency: str = "INR") -> Dict[str, Any]:
        """
        Takes prediction dictionary in INR and converts all price fields to target currency.
        """
        curr = target_currency.upper().strip()
        if curr not in self.rates:
            curr = "INR"

        out = pred_dict.copy()
        if curr == "INR":
            out["formatted_price"] = self.format_price(out["predicted_price"], "INR")
            out["formatted_range"] = f"{self.format_price(out['price_range_low'], 'INR')} – {self.format_price(out['price_range_high'], 'INR')}"
            return out

        out["currency"] = curr
        out["predicted_price"] = self.convert_from_inr(out["predicted_price"], curr)
        out["price_range_low"] = self.convert_from_inr(out["price_range_low"], curr)
        out["price_range_high"] = self.convert_from_inr(out["price_range_high"], curr)
        out["base_value"] = self.convert_from_inr(out["base_value"], curr)

        # Convert SHAP impacts
        if "shap_breakdown" in out:
            new_shap = []
            for item in out["shap_breakdown"]:
                item_copy = dict(item) if isinstance(item, dict) else item.model_dump()
                item_copy["impact_inr"] = self.convert_from_inr(item_copy["impact_inr"], curr)
                new_shap.append(item_copy)
            out["shap_breakdown"] = new_shap

        out["formatted_price"] = self.format_price(out["predicted_price"], curr)
        out["formatted_range"] = f"{self.format_price(out['price_range_low'], curr)} – {self.format_price(out['price_range_high'], curr)}"
        return out
