"""
Phase 11: Multi-Language (i18n) Translation Engine.
Translates vehicle attributes, SHAP feature breakdowns, and valuation report summaries
into English (en), Hindi (hi), Spanish (es), and Arabic (ar).
"""

from typing import Dict, Any


TRANSLATIONS = {
    "en": {
        "title": "Car Valuation Report",
        "estimated_price": "Estimated Fair Market Resale Value",
        "confidence_range": "Confidence Range",
        "feature_impact": "Feature Impact Breakdown (SHAP)",
        "car_age": "Car Age",
        "km_driven": "KM Driven",
        "owner_count": "Owner Count",
        "fuel_type": "Fuel Type",
        "brand_value": "Brand & Model Value",
        "positive": "Positive",
        "negative": "Negative",
    },
    "hi": {
        "title": "कार मूल्यांकन रिपोर्ट",
        "estimated_price": "अनुमानित उचित बाज़ार मूल्य",
        "confidence_range": "विश्वास सीमा (रेंज)",
        "feature_impact": "विशेषता प्रभाव विश्लेषण (SHAP)",
        "car_age": "कार की आयु",
        "km_driven": "चली गई दूरी (किमी)",
        "owner_count": "मालिकों की संख्या",
        "fuel_type": "ईंधन का प्रकार",
        "brand_value": "ब्रांड एवं मॉडल मूल्य",
        "positive": "सकारात्मक",
        "negative": "नकारात्मक",
    },
    "es": {
        "title": "Informe de Valoración de Vehículo",
        "estimated_price": "Valor Estimado de Mercado",
        "confidence_range": "Rango de Confianza",
        "feature_impact": "Desglose de Impacto de Características (SHAP)",
        "car_age": "Antigüedad del Coche",
        "km_driven": "Kilómetros Recorridos",
        "owner_count": "Número de Propietarios",
        "fuel_type": "Tipo de Combustible",
        "brand_value": "Valor de Marca y Modelo",
        "positive": "Positivo",
        "negative": "Negativo",
    },
    "ar": {
        "title": "تقرير تقييم السيارة",
        "estimated_price": "القيمة السوقية العادلة المقدرة",
        "confidence_range": "نطاق الثقة",
        "feature_impact": "تحليل تأثير الميزات (SHAP)",
        "car_age": "عمر السيارة",
        "km_driven": "المسافة المقطوعة",
        "owner_count": "عدد المالكين",
        "fuel_type": "نوع الوقود",
        "brand_value": "قيمة العلامة التجارية والطرّاز",
        "positive": "إيجابي",
        "negative": "سلبي",
    }
}


class TranslationEngine:
    """
    Internationalization (i18n) Engine for Multi-Language Output Rendering.
    """

    def __init__(self, default_lang: str = "en"):
        self.default_lang = default_lang

    def get_supported_languages(self) -> Dict[str, str]:
        return {
            "en": "English",
            "hi": "हिन्दी (Hindi)",
            "es": "Español (Spanish)",
            "ar": "العربية (Arabic)"
        }

    def translate_key(self, key: str, lang: str = "en") -> str:
        lang_code = lang.lower().strip() if lang else "en"
        lang_dict = TRANSLATIONS.get(lang_code, TRANSLATIONS["en"])
        return lang_dict.get(key, TRANSLATIONS["en"].get(key, key))

    def translate_shap_breakdown(self, breakdown: list, lang: str = "en") -> list:
        if not breakdown or lang.lower() == "en":
            return breakdown

        translated = []
        for item in breakdown:
            item_copy = dict(item) if isinstance(item, dict) else item.model_dump()
            feat_name = item_copy.get("feature", "")

            # Translate known feature names
            if "Age" in feat_name:
                item_copy["feature"] = self.translate_key("car_age", lang)
            elif "KM" in feat_name:
                item_copy["feature"] = self.translate_key("km_driven", lang)
            elif "Owner" in feat_name:
                item_copy["feature"] = self.translate_key("owner_count", lang)
            elif "Fuel" in feat_name:
                item_copy["feature"] = self.translate_key("fuel_type", lang)
            elif "Brand" in feat_name:
                item_copy["feature"] = self.translate_key("brand_value", lang)

            direction = item_copy.get("direction", "")
            if direction == "positive":
                item_copy["direction"] = self.translate_key("positive", lang)
            elif direction == "negative":
                item_copy["direction"] = self.translate_key("negative", lang)

            translated.append(item_copy)
        return translated
