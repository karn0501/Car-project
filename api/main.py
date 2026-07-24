"""
Phase 8: FastAPI Backend Application.
Production-grade REST API for Used Car Price Prediction with SHAP explainability,
CNN condition scoring, LSTM trend forecasting, comparable listings, PDF reports,
and user feedback collection.

Endpoints:
    POST /predict          — Price prediction with SHAP breakdown
    POST /upload-image     — Car condition scoring from photo
    GET  /trend/{variant}  — 3-month price trend forecast
    GET  /compare          — Find comparable listings
    POST /feedback         — Submit actual price feedback
    GET  /report/{pred_id} — Download PDF valuation report
    GET  /health           — Health check
"""

import os
import sys
import time
import uuid
from datetime import datetime
from collections import defaultdict
from io import BytesIO

# Add project root to python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends, Query
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    PredictRequest, PredictResponse, FeatureImpact,
    ImageScoreResponse, TrendResponse, ForecastPoint,
    CompareResponse, ComparableListing,
    FeedbackRequest, FeedbackResponse,
    HealthResponse,
)
from api.services import (
    PredictionService, ImageScoringService,
    TrendService, ComparisonService, FeedbackService,
)
from api.report_generator import generate_valuation_report
from api.auth import router as auth_router
from api.nlp_query_parser import CarQueryParser


# ─── App Configuration ────────────────────────────────────────────────────────

app = FastAPI(
    title="Used Car Price Prediction API",
    description="Production-grade ML/DL API for accurate used car valuation with "
                "SHAP explainability, CNN condition scoring, LSTM trend forecasting, "
                "comparable listings, and PDF valuation reports.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Include Auth Router
app.include_router(auth_router)

# Mount Static Files for UI Dashboard
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

query_parser = CarQueryParser()

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Rate Limiting Middleware ─────────────────────────────────────────────────

RATE_LIMIT = 60  # requests per minute per IP
rate_limit_store = defaultdict(list)

API_KEY = os.getenv("CAR_API_KEY", "car-prediction-api-key-2026")


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple IP-based rate limiter: 60 requests/minute."""
    # Skip rate limiting for docs and health
    if request.url.path in ("/docs", "/redoc", "/openapi.json", "/health"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Clean old entries (older than 60 seconds)
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip] if now - t < 60
    ]

    if len(rate_limit_store[client_ip]) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Max 60 requests per minute."}
        )

    rate_limit_store[client_ip].append(now)
    return await call_next(request)


# ─── API Key Authentication ───────────────────────────────────────────────────

def verify_api_key(request: Request):
    """Optional API key verification via X-API-Key header."""
    api_key = request.headers.get("X-API-Key")
    if api_key and api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    # If no key provided, allow access (public mode)
    return True


# ─── Service Initialization ──────────────────────────────────────────────────

prediction_service = PredictionService()
image_service = ImageScoringService()
trend_service = TrendService()
comparison_service = ComparisonService()
feedback_service = FeedbackService()

# Cache for predictions (for report generation)
prediction_cache = {}


# ─── Endpoints ────────────────────────────────────────────────────────────────

from api.metering import APIMeteringManager
from src.drift_detector import ModelDriftDetector

metering_manager = APIMeteringManager()
drift_detector = ModelDriftDetector()


@app.get("/metrics", tags=["Enterprise Monitoring"])
async def get_prometheus_metrics():
    """Prometheus-compatible system metrics endpoint."""
    total_preds = len(prediction_cache)
    return Response(
        content=f"# HELP car_predictions_total Total predictions generated\n"
                f"# TYPE car_predictions_total counter\n"
                f"car_predictions_total {total_preds}\n"
                f"# HELP car_api_status System operational status\n"
                f"# TYPE car_api_status gauge\n"
                f"car_api_status 1\n",
        media_type="text/plain"
    )


@app.get("/api/usage", tags=["B2B Metering"])
async def get_api_usage(request: Request):
    """Query current B2B API key request metrics & quota remaining."""
    api_key = request.headers.get("X-API-Key", "car-prediction-api-key-2026")
    metrics = metering_manager.get_key_metrics(api_key)
    if not metrics:
        raise HTTPException(status_code=404, detail="API key metrics not found")
    return metrics


@app.post("/drift/check", tags=["Enterprise Monitoring"])
async def check_data_drift():
    """Triggers statistical data drift analysis against baseline dataset."""
    try:
        from db.data_exporter import load_dataset_from_db
        live_df = load_dataset_from_db()
        report = drift_detector.evaluate_dataset_health(live_df)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift check failed: {str(e)}")


from src.currency_engine import CurrencyConverter
from src.regional_tax import RegionalTaxCalculator
from src.i18n import TranslationEngine

currency_converter = CurrencyConverter()
regional_tax_calculator = RegionalTaxCalculator()
translation_engine = TranslationEngine()


@app.get("/currencies", tags=["Multi-Region Localization"])
async def get_supported_currencies():
    """Returns supported currency symbols and translation languages."""
    return {
        "currencies": currency_converter.get_supported_currencies(),
        "languages": translation_engine.get_supported_languages()
    }


@app.post("/predict/localized", tags=["Multi-Region Localization"])
async def predict_localized(request: PredictRequest,
                            currency: str = Query(default="INR", description="Target currency (INR, USD, EUR, GBP, AED, JPY)"),
                            lang: str = Query(default="en", description="Output language (en, hi, es, ar)"),
                            _=Depends(verify_api_key)):
    """
    Predict price localized with target currency, regional RTO tax calculations,
    and multi-language SHAP explainability translations.
    """
    car_data = request.model_dump()
    raw_res = prediction_service.predict(car_data)

    # 1. Currency Conversion
    localized_res = currency_converter.localize_prediction_prices(raw_res, currency)

    # 2. Regional RTO Tax Calculation
    regional_fees = regional_tax_calculator.calculate_regional_fees(
        city=car_data.get("city", "Mumbai"),
        base_price=raw_res["predicted_price"],
        fuel_type=car_data.get("fuel_type", "Petrol"),
        car_age=datetime.now().year - car_data.get("manufacture_year", 2020)
    )

    # Convert regional fees to target currency if needed
    if currency.upper() != "INR":
        regional_fees["base_vehicle_price"] = currency_converter.convert_from_inr(regional_fees["base_vehicle_price"], currency)
        regional_fees["estimated_rto_tax"] = currency_converter.convert_from_inr(regional_fees["estimated_rto_tax"], currency)
        regional_fees["ncr_diesel_penalty"] = currency_converter.convert_from_inr(regional_fees["ncr_diesel_penalty"], currency)
        regional_fees["transfer_admin_fee"] = currency_converter.convert_from_inr(regional_fees["transfer_admin_fee"], currency)
        regional_fees["total_regional_cost"] = currency_converter.convert_from_inr(regional_fees["total_regional_cost"], currency)
        regional_fees["total_buyer_landed_cost"] = currency_converter.convert_from_inr(regional_fees["total_buyer_landed_cost"], currency)

    # 3. Multi-language Translation
    if lang and lang != "en":
        localized_res["shap_breakdown"] = translation_engine.translate_shap_breakdown(
            localized_res["shap_breakdown"], lang
        )

    localized_res["regional_tax_breakdown"] = regional_fees
    localized_res["language"] = lang

    return localized_res


@app.get("/", tags=["UI Dashboard"])
async def root_index():
    """Serves the interactive web application dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AutoValuate AI System API Online. Visit /docs for API documentation."}


@app.post("/chat-predict", tags=["Conversational AI"])
async def chat_predict(body: dict):
    """
    Accepts natural language text query (e.g. '2021 Hyundai Creta SX Diesel automatic 25000 km in Delhi'),
    parses entities, and returns instant price prediction.
    """
    query_text = body.get("query", "")
    parsed_attrs = query_parser.parse_query(query_text)
    prediction_result = prediction_service.predict(parsed_attrs)

    return {
        "query": query_text,
        "parsed": parsed_attrs,
        "prediction": prediction_result
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint — returns system status and model availability."""
    try:
        from db.data_exporter import load_dataset_from_db
        df = load_dataset_from_db()
        db_status = f"connected ({len(df)} listings)"
    except Exception:
        db_status = "fallback (SQLite)"

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "ensemble": prediction_service.is_loaded(),
            "cnn_condition": image_service.is_loaded(),
            "lstm_trend": trend_service.is_loaded(),
        },
        database_status=db_status,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_price(request: PredictRequest, _=Depends(verify_api_key)):
    """
    Predict the resale price of a used car.

    Returns predicted price, confidence range (10th-90th percentile),
    and SHAP feature impact breakdown for explainability.
    """
    car_data = request.model_dump()
    result = prediction_service.predict(car_data)

    # Cache prediction for report generation
    prediction_cache[result["prediction_id"]] = {
        "prediction": result,
        "car_data": car_data,
    }

    return PredictResponse(
        predicted_price=result["predicted_price"],
        price_range_low=result["price_range_low"],
        price_range_high=result["price_range_high"],
        currency=result["currency"],
        shap_breakdown=[FeatureImpact(**item) for item in result["shap_breakdown"]],
        base_value=result["base_value"],
        description_quality_score=result.get("description_quality_score"),
        prediction_id=result["prediction_id"],
        timestamp=result["timestamp"],
    )


@app.post("/upload-image", response_model=ImageScoreResponse, tags=["Computer Vision"])
async def upload_image(file: UploadFile = File(...), _=Depends(verify_api_key)):
    """
    Upload a car photo for AI-powered condition assessment.

    Returns predicted damage tier (0-3), condition label, visual score,
    and confidence probabilities across all tiers.
    """
    try:
        from PIL import Image

        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        result = image_service.score_image(image)

        return ImageScoreResponse(
            predicted_tier=result["predicted_tier"],
            condition_label=result["condition_label"],
            visual_condition_score=result["visual_condition_score"],
            confidence_probabilities=result.get("confidence_probabilities", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")


@app.get("/trend/{variant_key}", response_model=TrendResponse, tags=["Forecasting"])
async def get_price_trend(variant_key: str, months: int = Query(default=3, ge=1, le=12),
                           _=Depends(verify_api_key)):
    """
    Get LSTM-based price trend forecast for a specific car variant.

    Returns predicted price trajectory over the next N months with
    macroeconomic signal integration (fuel prices, interest rates, festive seasons).
    """
    result = trend_service.forecast(variant_key, months_ahead=months)

    if result.get("status") != "success":
        # Return demo forecast
        return TrendResponse(
            variant_key=variant_key,
            trend_direction="STABLE",
            base_price=500000,
            forecasts=[
                ForecastPoint(month=i + 1, date=f"2026-{8 + i:02d}",
                              predicted_price=500000 * (1 - 0.01 * (i + 1)),
                              change_pct=round(-1.0 * (i + 1), 2))
                for i in range(months)
            ],
        )

    return TrendResponse(
        variant_key=variant_key,
        trend_direction=result["trend_direction"],
        base_price=result["base_price"],
        forecasts=[ForecastPoint(**fc) for fc in result["forecasts"]],
    )


@app.get("/compare", response_model=CompareResponse, tags=["Comparable Listings"])
async def compare_listings(
    company: str = Query(..., description="Car manufacturer"),
    model: str = Query(..., description="Car model"),
    year: int = Query(..., ge=1990, le=2027, description="Manufacture year"),
    city: str = Query(default=None, description="Optional city filter"),
    limit: int = Query(default=5, ge=1, le=10, description="Max results"),
    _=Depends(verify_api_key),
):
    """
    Find comparable listings from the database.

    Returns 3-5 real, similar listings that users can use to
    sanity-check the predicted price.
    """
    listings = comparison_service.find_comparable(company, model, year, city, limit)

    return CompareResponse(
        query_summary=f"{company} {model} ({year})" + (f" in {city}" if city else ""),
        comparable_count=len(listings),
        listings=[ComparableListing(**l) for l in listings],
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest, _=Depends(verify_api_key)):
    """
    Submit feedback on the actual sale/purchase price.

    This ground-truth data is used for future model retraining
    and measuring real-world prediction accuracy.
    """
    result = feedback_service.submit_feedback(
        prediction_id=request.prediction_id,
        actual_price=request.actual_price,
        comments=request.comments,
    )

    return FeedbackResponse(**result)


@app.get("/report/{prediction_id}", tags=["Reports"])
async def download_report(prediction_id: str, _=Depends(verify_api_key)):
    """
    Download a PDF valuation report for a previous prediction.

    Contains predicted price/range, SHAP breakdown, comparable listings,
    and price trend forecast.
    """
    cached = prediction_cache.get(prediction_id)
    if not cached:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found. Make a prediction first.")

    prediction_data = cached["prediction"]
    car_data = cached["car_data"]

    # Get comparable listings
    comparables = comparison_service.find_comparable(
        company=car_data.get("company_name", ""),
        model=car_data.get("model_name", ""),
        year=car_data.get("manufacture_year", 2020),
        city=car_data.get("city"),
        limit=5,
    )

    # Get trend data
    variant_key = f"{car_data.get('company_name', '')} {car_data.get('model_name', '')}"
    trend_data = trend_service.forecast(variant_key, months_ahead=3)

    # Generate report
    report_bytes = generate_valuation_report(prediction_data, comparables, trend_data)

    # Determine content type
    content_type = "application/pdf" if report_bytes[:4] == b"%PDF" else "text/plain"
    filename = f"valuation_report_{prediction_id}.pdf" if content_type == "application/pdf" else f"valuation_report_{prediction_id}.txt"

    return Response(
        content=report_bytes,
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ─── Startup Event ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Log startup status."""
    print("=" * 70)
    print("Used Car Price Prediction API — Starting Up")
    print("=" * 70)
    print(f"  |-- Ensemble Model  : {'LOADED' if prediction_service.is_loaded() else 'NOT LOADED'}")
    print(f"  |-- CNN Condition   : {'LOADED' if image_service.is_loaded() else 'NOT LOADED'}")
    print(f"  |-- LSTM Trend      : {'LOADED' if trend_service.is_loaded() else 'NOT LOADED'}")
    print(f"  \\-- API Docs        : http://localhost:8000/docs")
    print("=" * 70)
