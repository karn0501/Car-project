"""
Phase 8: Pydantic Request/Response Schemas for FastAPI endpoints.
Defines strict, validated data models for all API interactions.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


# ─── Prediction Schemas ───────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Input schema for car price prediction."""
    company_name: str = Field(..., description="Car manufacturer (e.g., 'Maruti', 'Hyundai')")
    model_name: str = Field(..., description="Car model (e.g., 'Swift', 'Creta')")
    variant_name: str = Field(default="Base", description="Variant name (e.g., 'VXi', 'SX')")
    manufacture_year: int = Field(..., ge=1990, le=2027, description="Year of manufacture")
    km_driven: float = Field(..., ge=0, le=500000, description="Kilometers driven")
    fuel_type: str = Field(default="Petrol", description="Fuel type")
    transmission: str = Field(default="Manual", description="Transmission type")
    owner_count: int = Field(default=1, ge=1, le=5, description="Number of previous owners")
    city: str = Field(default="Mumbai", description="City of registration")
    body_type: str = Field(default="Hatchback", description="Body type")
    engine_cc: float = Field(default=1200.0, ge=500, le=6000, description="Engine displacement in cc")
    seating_capacity: int = Field(default=5, ge=2, le=10, description="Seating capacity")
    insurance_valid: str = Field(default="Yes", description="Insurance validity")
    accident_history: str = Field(default="No", description="Accident history")
    description: Optional[str] = Field(default=None, description="Listing description text for NLP scoring")

    class Config:
        json_schema_extra = {
            "example": {
                "company_name": "Maruti",
                "model_name": "Swift",
                "variant_name": "VXi",
                "manufacture_year": 2020,
                "km_driven": 35000,
                "fuel_type": "Petrol",
                "transmission": "Manual",
                "owner_count": 1,
                "city": "Mumbai",
                "body_type": "Hatchback",
                "engine_cc": 1197,
                "seating_capacity": 5,
                "insurance_valid": "Yes",
                "accident_history": "No",
            }
        }


class FeatureImpact(BaseModel):
    """Single feature contribution to predicted price."""
    feature: str
    impact_inr: float
    direction: str  # "positive" or "negative"


class PredictResponse(BaseModel):
    """Output schema for car price prediction."""
    predicted_price: float
    price_range_low: float
    price_range_high: float
    currency: str = "INR"
    shap_breakdown: List[FeatureImpact]
    base_value: float
    description_quality_score: Optional[float] = None
    prediction_id: str
    timestamp: str


# ─── Image Scoring Schemas ────────────────────────────────────────────────────

class ImageScoreResponse(BaseModel):
    """Output schema for car condition image scoring."""
    predicted_tier: int
    condition_label: str
    visual_condition_score: float
    confidence_probabilities: Dict[str, float]


# ─── Trend Forecast Schemas ───────────────────────────────────────────────────

class ForecastPoint(BaseModel):
    """Single month forecast data point."""
    month: int
    date: str
    predicted_price: float
    change_pct: float


class TrendResponse(BaseModel):
    """Output schema for price trend forecast."""
    variant_key: str
    trend_direction: str
    base_price: float
    forecasts: List[ForecastPoint]


# ─── Comparable Listings Schemas ──────────────────────────────────────────────

class ComparableListing(BaseModel):
    """A single comparable listing from the database."""
    company_name: str
    model_name: str
    variant_name: str
    manufacture_year: int
    km_driven: float
    asking_price: float
    city: str
    fuel_type: str = ""
    transmission: str = ""


class CompareResponse(BaseModel):
    """Output schema for comparable listings."""
    query_summary: str
    comparable_count: int
    listings: List[ComparableListing]


# ─── Feedback Schemas ─────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """User feedback on actual sale/purchase price."""
    prediction_id: str = Field(..., description="ID from the prediction response")
    actual_price: float = Field(..., gt=0, description="Actual sale/purchase price in INR")
    comments: Optional[str] = Field(default=None, description="Optional user comments")


class FeedbackResponse(BaseModel):
    """Confirmation of feedback submission."""
    status: str
    feedback_id: str
    message: str


# ─── Health Check ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    database_status: str
    timestamp: str
