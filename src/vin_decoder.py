"""
Phase 12: 17-Character VIN / Chassis Spec Decoder.
Decodes ISO 3779 17-character Vehicle Identification Numbers (VIN) to extract
manufacturer (WMI), model year (10th digit), assembly plant, and vehicle specifications.
"""

from typing import Dict, Any


# World Manufacturer Identifier (WMI) Map (First 3 Characters)
WMI_MAP = {
    "MA3": {"company": "Maruti", "country": "India"},
    "MBH": {"company": "Hyundai", "country": "India"},
    "MAT": {"company": "Tata", "country": "India"},
    "MA1": {"company": "Mahindra", "country": "India"},
    "MAK": {"company": "Honda", "country": "India"},
    "MBJ": {"company": "Toyota", "country": "India"},
    "MZB": {"company": "Kia", "country": "India"},
    "WVW": {"company": "Volkswagen", "country": "Germany"},
    "TMB": {"company": "Skoda", "country": "Czech Republic"},
    "1FA": {"company": "Ford", "country": "USA"},
}

# 10th Digit Model Year Code Map
VIN_YEAR_MAP = {
    'J': 2018, 'K': 2019, 'L': 2020, 'M': 2021, 'N': 2022,
    'P': 2023, 'R': 2024, 'S': 2025, 'T': 2026, 'V': 2027
}


class VINDecoder:
    """
    17-Character ISO 3779 VIN / Chassis Number Decoder Engine.
    """

    def decode_vin(self, vin: str) -> Dict[str, Any]:
        if not vin or not isinstance(vin, str):
            return {"valid": False, "reason": "Empty VIN string"}

        clean_vin = vin.strip().upper().replace(" ", "").replace("-", "")

        if len(clean_vin) != 17:
            return {
                "valid": False,
                "reason": f"Invalid VIN length ({len(clean_vin)} chars). Must be exactly 17 characters.",
                "vin": clean_vin
            }

        # 1. WMI (First 3 chars)
        wmi_code = clean_vin[:3]
        wmi_info = WMI_MAP.get(wmi_code, {"company": "Unknown", "country": "Global"})

        # 2. Model Year (10th char)
        year_char = clean_vin[9]
        manufacture_year = VIN_YEAR_MAP.get(year_char, 2021)

        # 3. Serial / Sequence number (Last 6 chars)
        serial_number = clean_vin[11:]

        # 4. Assembly Plant Code (11th char)
        plant_code = clean_vin[10]

        return {
            "valid": True,
            "vin": clean_vin,
            "wmi_code": wmi_code,
            "company_name": wmi_info["company"],
            "country_of_origin": wmi_info["country"],
            "manufacture_year": manufacture_year,
            "year_code": year_char,
            "plant_code": plant_code,
            "serial_number": serial_number,
            "body_type": "Sedan/SUV/Hatchback",
            "fuel_type": "Petrol/Diesel"
        }
