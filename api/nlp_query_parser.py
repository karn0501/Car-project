"""
Phase 9: Natural Language Car Query Parser.
Parses free-text user queries into structured car attributes for instant conversational AI pricing.
Example input: "2021 Hyundai Creta SX Diesel automatic 25,000 km in Delhi"
Returns: {
    "company_name": "Hyundai",
    "model_name": "Creta",
    "variant_name": "SX",
    "manufacture_year": 2021,
    "km_driven": 25000,
    "fuel_type": "Diesel",
    "transmission": "Automatic",
    "city": "Delhi"
}
"""

import re
from typing import Dict, Any


KNOWN_COMPANIES = [
    "maruti", "maruti suzuki", "hyundai", "tata", "mahindra", "honda", "toyota",
    "kia", "volkswagen", "skoda", "renault", "nissan", "ford", "mg", "bmw", "mercedes"
]

KNOWN_CITIES = [
    "mumbai", "delhi", "bangalore", "bengaluru", "chennai", "hyderabad", "pune",
    "ahmedabad", "kolkata", "jaipur", "chandigarh", "lucknow", "kochi", "indore"
]

FUEL_TYPES = {
    "petrol": "Petrol",
    "diesel": "Diesel",
    "cng": "CNG",
    "electric": "Electric",
    "ev": "Electric",
    "hybrid": "Hybrid"
}

TRANSMISSIONS = {
    "automatic": "Automatic",
    "auto": "Automatic",
    "amt": "Automatic",
    "cvt": "Automatic",
    "dct": "Automatic",
    "manual": "Manual",
    "mt": "Manual"
}


class CarQueryParser:
    """
    Regex and Rule-based Natural Language Entity Extractor for Used Car Queries.
    """

    def parse_query(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return self._default_attributes()

        raw_text = text.strip()
        lower_text = raw_text.lower()

        # 1. Manufacture Year (4-digit number between 1990 and 2027)
        year_match = re.search(r"\b(199\d|200\d|201\d|202[0-7])\b", lower_text)
        year = int(year_match.group(1)) if year_match else 2020

        # 2. KM Driven (e.g., 25000, 25k, 25,000 km, 45000 kms)
        km_driven = 35000.0
        # First try explicit km markers
        km_match = re.search(r"(\d+[\d,]*)\s*(k|km|kms|kilometers)\b", lower_text)
        if not km_match:
            # Fallback to numbers other than year
            all_nums = re.findall(r"\b(\d+[\d,]*)\b", lower_text)
            for n_str in all_nums:
                clean_n = n_str.replace(",", "")
                if clean_n.isdigit():
                    v = float(clean_n)
                    if v != year and 1000 <= v <= 600000:
                        km_driven = v
                        break
        else:
            val_str = km_match.group(1).replace(",", "")
            if val_str.isdigit():
                val = float(val_str)
                if km_match.group(2) == "k":
                    val *= 1000
                if val != year and val <= 600000:
                    km_driven = val

        # 3. Fuel Type
        fuel = "Petrol"
        for kw, canonical in FUEL_TYPES.items():
            if re.search(r"\b" + kw + r"\b", lower_text):
                fuel = canonical
                break

        # 4. Transmission
        transmission = "Manual"
        for kw, canonical in TRANSMISSIONS.items():
            if re.search(r"\b" + kw + r"\b", lower_text):
                transmission = canonical
                break

        # 5. City
        city = "Mumbai"
        for c in KNOWN_CITIES:
            if re.search(r"\b" + c + r"\b", lower_text):
                city = c.capitalize()
                if city == "Bengaluru":
                    city = "Bangalore"
                break

        # 6. Company & Model extraction
        company = "Maruti"
        model = "Swift"
        variant = "VXi"

        for comp in KNOWN_COMPANIES:
            if comp in lower_text:
                company = comp.title()
                if company.lower() == "maruti suzuki":
                    company = "Maruti"

                # Look for model word after company
                pattern = r"\b" + re.escape(comp) + r"\s+([a-z0-9\-]+)"
                m_match = re.search(pattern, lower_text)
                if m_match:
                    possible_model = m_match.group(1).capitalize()
                    if possible_model.lower() not in ["2018", "2019", "2020", "2021", "2022", "petrol", "diesel"]:
                        model = possible_model
                break

        # Check common standalone model names
        if "creta" in lower_text:
            company, model = "Hyundai", "Creta"
        elif "swift" in lower_text:
            company, model = "Maruti", "Swift"
        elif "baleno" in lower_text:
            company, model = "Maruti", "Baleno"
        elif "city" in lower_text:
            company, model = "Honda", "City"
        elif "nexon" in lower_text:
            company, model = "Tata", "Nexon"
        elif "seltos" in lower_text:
            company, model = "Kia", "Seltos"
        elif "fortuner" in lower_text:
            company, model = "Toyota", "Fortuner"
        elif "i20" in lower_text:
            company, model = "Hyundai", "i20"

        # Check variant keywords
        if "vxi" in lower_text:
            variant = "VXi"
        elif "zxi" in lower_text:
            variant = "ZXi"
        elif "lxi" in lower_text:
            variant = "LXi"
        elif "sx" in lower_text:
            variant = "SX"
        elif "xz" in lower_text:
            variant = "XZ"
        elif "vx" in lower_text:
            variant = "VX"

        return {
            "company_name": company,
            "model_name": model,
            "variant_name": variant,
            "manufacture_year": year,
            "km_driven": km_driven,
            "fuel_type": fuel,
            "transmission": transmission,
            "owner_count": 1,
            "city": city,
            "body_type": "Hatchback" if model in ["Swift", "Baleno", "i20"] else ("SUV" if model in ["Creta", "Nexon", "Seltos", "Fortuner"] else "Sedan"),
            "engine_cc": 1197.0 if company == "Maruti" else 1497.0,
            "seating_capacity": 5,
            "insurance_valid": "Yes",
            "accident_history": "No",
            "original_query": raw_text
        }

    def _default_attributes(self) -> Dict[str, Any]:
        return {
            "company_name": "Maruti",
            "model_name": "Swift",
            "variant_name": "VXi",
            "manufacture_year": 2020,
            "km_driven": 35000.0,
            "fuel_type": "Petrol",
            "transmission": "Manual",
            "owner_count": 1,
            "city": "Mumbai",
            "body_type": "Hatchback",
            "engine_cc": 1197.0,
            "seating_capacity": 5,
            "insurance_valid": "Yes",
            "accident_history": "No",
            "original_query": ""
        }
