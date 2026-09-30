"""
AgriLink Unified Backend API — FastAPI
Standardized, AI-powered digital agriculture platform for crop advisory, weather, soil,
satellite environmental metrics, disease detection, interoperability, and localization.
"""

import os
import io
import json
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Internal services & database
from backend.services.weather import fetch_weather
from backend.services.soil import get_soil_data
from backend.services.satellite import get_satellite_data
from backend.services.disease import analyze_crop_image
from backend.services.advisory import generate_ai_advisory, generate_ai_chat_response, generate_structured_context, GOOGLE_CLOUD_GEMINI_MODEL
from backend.services.interop import (
    AGRIDATA_SCHEMA_SPEC,
    build_standard_record,
    validate_agridata_payload,
    get_sample_crossborder_records
)
from backend.utils.localization import get_translation_bundle, TRANSLATIONS, CROP_NAMES_TRANSLATION
from backend.train_model import generate_explainable_reasons, get_crop_compatibility_breakdown
from backend.db.database import (
    init_db,
    list_farms,
    save_farm,
    log_advisory,
    log_disease_scan,
    get_recent_history
)

# Initialize database on startup
init_db()

app = FastAPI(
    title="AgriLink Platform API",
    description="Interoperable AI-powered agricultural advisory, disease detection, and telemetry engine.",
    version="1.0.0"
)

# Enable CORS for frontend client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_FILE = next(
    (os.path.join(PROJECT_ROOT, f) for f in ["index.html", "frontend_index.html", "frontend/index.html"] if os.path.exists(os.path.join(PROJECT_ROOT, f))),
    os.path.join(PROJECT_ROOT, "frontend_index.html")
)
FRONTEND_ASSETS = next(
    (os.path.join(PROJECT_ROOT, d) for d in ["assets", "frontend/assets"] if os.path.isdir(os.path.join(PROJECT_ROOT, d))),
    os.path.join(PROJECT_ROOT, "frontend", "assets")
)
if os.path.isdir(FRONTEND_ASSETS):
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="assets")

# Load Trained Crop Model Bundle
MODEL_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "crop_model.pkl"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_model.pkl"),
    os.path.join(PROJECT_ROOT, "backend", "models", "crop_model.pkl"),
    os.path.join(PROJECT_ROOT, "backend", "crop_model.pkl"),
    os.path.join(PROJECT_ROOT, "crop_model.pkl")
]
_model_bundle = None



def get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        for path in MODEL_PATHS:
            if os.path.exists(path):
                try:
                    _model_bundle = joblib.load(path)
                    break
                except Exception as e:
                    print(f"Failed loading model from {path}: {e}")
        if _model_bundle is None:
            raise HTTPException(
                status_code=500,
                detail="Trained crop model not found. Run `python backend/train_model.py` first."
            )
    return _model_bundle


# ==========================================
# PYDANTIC SCHEMAS
# ==========================================

class CropRecommendationRequest(BaseModel):
    nitrogen: float = Field(..., ge=0, le=350, description="Soil Nitrogen (N, kg/ha)")
    phosphorus: float = Field(..., ge=0, le=250, description="Soil Phosphorus (P, kg/ha)")
    potassium: float = Field(..., ge=0, le=300, description="Soil Potassium (K, kg/ha)")
    temperature: float = Field(..., ge=-10, le=60, description="Ambient Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Relative Humidity (%)")
    ph: float = Field(..., ge=0, le=14, description="Soil pH level")
    rainfall: float = Field(..., ge=0, le=2000, description="Seasonal Rainfall (mm)")
    soil_moisture: Optional[float] = Field(0.35, ge=0, le=1.0, description="Soil moisture fraction")
    farm_id: Optional[str] = Field(None, description="Associated Farm ID")
    language: Optional[str] = Field("en", description="Target localization (en, te, hi)")


class FarmCreateRequest(BaseModel):
    id: Optional[str] = None
    name: str = Field(..., description="Farm or Field Name")
    farmer_name: Optional[str] = "Farmer"
    region: str = Field("Telangana", description="Administrative Region or State")
    latitude: float = Field(17.3850, description="Farm Latitude")
    longitude: float = Field(78.4867, description="Farm Longitude")
    area_acres: Optional[float] = 5.0
    soil_type: Optional[str] = "Black Cotton Soil"


class AIAdvisoryRequest(BaseModel):
    farm_info: Dict[str, Any]
    weather_data: Dict[str, Any]
    soil_data: Dict[str, Any]
    satellite_data: Dict[str, Any]
    crop_recommendation: Dict[str, Any]
    disease_result: Optional[Dict[str, Any]] = None
    language: Optional[str] = "en"


class AIChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=2000)


class AIChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1200)
    farm_info: Dict[str, Any]
    weather_data: Dict[str, Any]
    soil_data: Dict[str, Any]
    satellite_data: Dict[str, Any]
    crop_recommendation: Optional[Dict[str, Any]] = None
    disease_result: Optional[Dict[str, Any]] = None
    history: List[AIChatTurn] = Field(default_factory=list)
    language: Optional[str] = "en"


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AgriLink Digital Agriculture API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoints": {
            "docs": "/docs",
            "weather": "/api/weather",
            "soil": "/api/soil",
            "satellite": "/api/satellite",
            "crop_recommendation": "/api/crop-recommendation",
            "disease_diagnostic": "/api/disease",
            "ai_advisor": "/api/ai-advisory",
            "interoperability_schema": "/api/schema",
            "farms": "/api/farms",
            "history": "/api/history",
            "locales": "/api/locales"
        }
    }


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_FILE, media_type="text/html")


# ---------- 1. WEATHER SERVICE ----------
@app.get("/api/weather")
async def get_weather(lat: float = Query(17.3850), lon: float = Query(78.4867)):
    """Fetches live meteorological conditions and 7-day forecast from Open-Meteo."""
    return await fetch_weather(lat=lat, lon=lon)


# ---------- 2. SOIL SERVICE ----------
@app.get("/api/soil")
def get_soil(
    region: Optional[str] = Query("Telangana"),
    ph: Optional[float] = None,
    nitrogen: Optional[float] = None,
    phosphorus: Optional[float] = None,
    potassium: Optional[float] = None,
    moisture: Optional[float] = None
):
    """Retrieves regional soil profile with health thresholds, or incorporates user-provided overrides."""
    user_overrides = {}
    if ph is not None: user_overrides["ph"] = ph
    if nitrogen is not None: user_overrides["nitrogen"] = nitrogen
    if phosphorus is not None: user_overrides["phosphorus"] = phosphorus
    if potassium is not None: user_overrides["potassium"] = potassium
    if moisture is not None: user_overrides["moisture"] = moisture

    return get_soil_data(region=region, user_data=user_overrides if user_overrides else None)


# ---------- 3. SATELLITE & REMOTE SENSING ----------
@app.get("/api/satellite")
def get_satellite(lat: float = Query(17.3850), lon: float = Query(78.4867), region: Optional[str] = Query("Telangana")):
    """Retrieves remote sensing vegetation indices (NDVI, soil moisture, LST) with clear data provenance."""
    return get_satellite_data(lat=lat, lon=lon, region=region)


# ---------- 4. CROP RECOMMENDATION ENGINE ----------
@app.post("/api/crop-recommendation")
@app.post("/api/advisory")
def crop_recommendation_endpoint(req: CropRecommendationRequest):
    """
    ML-driven crop suitability classifier with multi-factor explainability.
    Considers Soil N, P, K, pH, Temperature, Humidity, Rainfall, and Root-zone Moisture.
    """
    bundle = get_model_bundle()
    model = bundle["model"]
    regen_tips = bundle.get("regen_tips", {})
    features = bundle.get("features", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_moisture"])

    input_map = {
        "N": req.nitrogen,
        "P": req.phosphorus,
        "K": req.potassium,
        "temperature": req.temperature,
        "humidity": req.humidity,
        "ph": req.ph,
        "rainfall": req.rainfall,
        "soil_moisture": req.soil_moisture or 0.35,
    }

    feature_vector = np.array([[input_map.get(f, 0.0) for f in features]], dtype=np.float32)
    probs = model.predict_proba(feature_vector)[0]
    classes = model.classes_
    order = np.argsort(probs)[::-1]

    top_crop = str(classes[order[0]])
    top_conf = float(probs[order[0]])

    # Generate explainable reasons comparing user input with crop agronomic ranges
    input_params_dict = {
        "ph": req.ph,
        "temperature": req.temperature,
        "rainfall": req.rainfall,
        "humidity": req.humidity,
        "nitrogen": req.nitrogen,
        "phosphorus": req.phosphorus,
        "potassium": req.potassium,
        "soil_moisture": req.soil_moisture or 0.35
    }
    reasons = generate_explainable_reasons(top_crop, input_params_dict)

    top_alternatives = []
    for i in order[1:4]:
        alt_name = str(classes[i])
        alt_conf = round(float(probs[i]), 3)
        alt_breakdown = get_crop_compatibility_breakdown(alt_name, input_params_dict)
        top_alternatives.append({
            "crop": alt_name,
            "confidence": alt_conf,
            "crop_localized": CROP_NAMES_TRANSLATION.get(alt_name, {}).get(req.language or "en", alt_name.capitalize()),
            "breakdown": alt_breakdown
        })

    regen_tip = regen_tips.get(top_crop, "Adopt cover cropping and conservation tillage to regenerate soil biology.")

    # Localized crop title
    crop_localized = CROP_NAMES_TRANSLATION.get(top_crop, {}).get(req.language or "en", top_crop.capitalize())

    # Log to DB if farm_id is present
    if req.farm_id:
        log_advisory(
            farm_id=req.farm_id,
            recommended_crop=top_crop,
            confidence=round(top_conf, 3),
            reasons=reasons,
            advisory_text=regen_tip,
            language=req.language or "en"
        )

    return {
        "recommended_crop": top_crop,
        "crop_localized": crop_localized,
        "confidence": round(top_conf, 3),
        "reasons": reasons,
        "top_alternatives": top_alternatives,
        "regenerative_tip": regen_tip,
        "input_echo": req.dict(),
        "model_version": "RandomForest-v2.1-Agronomic"
    }


# ---------- 5. CROP DISEASE DETECTION ----------
@app.post("/api/disease")
async def disease_diagnostic_endpoint(file: UploadFile = File(...), farm_id: Optional[str] = Query(None)):
    """Vision-based crop leaf disease classifier with visual symptom metrics and safety disclaimers."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid leaf image file (JPEG/PNG).")

    contents = await file.read()
    try:
        result = analyze_crop_image(contents, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed processing image: {e}")

    # Log scan to DB
    log_disease_scan(
        farm_id=farm_id,
        image_name=file.filename or "leaf_upload.jpg",
        crop_name=result.get("crop", "General Crop"),
        disease_name=result.get("diagnosis", "Unknown"),
        confidence=result.get("confidence", 0.0),
        severity=result.get("severity", "Moderate"),
        action="; ".join(result.get("recommended_actions", []))
    )

    return result


# ---------- 6. AI AGRICULTURAL ADVISOR ----------
@app.post("/api/ai-advisory")
async def ai_advisory_endpoint(req: AIAdvisoryRequest):
    """Synthesizes verified sensor, soil, weather, crop, and disease data into a cohesive advisory."""
    return await generate_ai_advisory(
        farm_info=req.farm_info,
        weather_data=req.weather_data,
        soil_data=req.soil_data,
        satellite_data=req.satellite_data,
        crop_recommendation=req.crop_recommendation,
        disease_result=req.disease_result,
        language=req.language or "en"
    )


@app.post("/api/ai-chat")
async def ai_chat_endpoint(req: AIChatRequest):
    """Answers a farmer's specific question with Vertex AI Gemini and verified farm context."""
    farm_info = {
        **req.farm_info,
        "latitude": req.farm_info.get("latitude", req.farm_info.get("lat", 17.3850)),
        "longitude": req.farm_info.get("longitude", req.farm_info.get("lon", 78.4867)),
        "area_acres": req.farm_info.get("area_acres", req.farm_info.get("area", 4.5)),
        "soil_type": req.farm_info.get("soil_type", req.farm_info.get("soilType", "Black Cotton Soil"))
    }
    weather_data = {
        **req.weather_data,
        "current": req.weather_data.get("current", req.weather_data)
    }
    soil_data = {
        **req.soil_data,
        "soil_type": req.soil_data.get("soil_type", req.soil_data.get("soilType", "Black Cotton Soil"))
    }
    context = generate_structured_context(
        farm_info=farm_info,
        weather_data=weather_data,
        soil_data=soil_data,
        satellite_data=req.satellite_data,
        crop_recommendation=req.crop_recommendation or {},
        disease_result=req.disease_result,
        language=req.language or "en"
    )
    context["weather_forecast"] = req.weather_data.get("forecast", [])
    context["farm"]["current_crop"] = req.farm_info.get("currentCrop", req.farm_info.get("current_crop"))

    try:
        answer = await generate_ai_chat_response(
            question=req.question.strip(),
            context=context,
            history=[turn.model_dump() for turn in req.history[-8:]],
            language=req.language or "en"
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {"answer": answer, "engine": f"Google Cloud Vertex AI ({GOOGLE_CLOUD_GEMINI_MODEL})"}


# ---------- 7. INTEROPERABILITY & STANDARDS ----------
@app.get("/api/schema")
def get_interoperability_schema():
    """Returns the standardized AgriData JSON schema specification."""
    return AGRIDATA_SCHEMA_SPEC


@app.post("/api/interop/export")
def export_standard_agridata(
    country: str = "IND",
    region: str = "Telangana",
    lat: float = 17.3850,
    lon: float = 78.4867,
    soil_data: Dict[str, Any] = None,
    weather_data: Dict[str, Any] = None,
    satellite_data: Dict[str, Any] = None,
    advisory_data: Optional[Dict[str, Any]] = None
):
    """Constructs a compliant, exportable AgriData cross-border exchange record."""
    soil = soil_data or get_soil_data(region=region)
    weather = weather_data or {"current": {"temperature": 29.0, "humidity": 70.0, "precipitation": 0.0, "wind_speed": 12.0, "condition": "Mainly clear"}}
    satellite = satellite_data or get_satellite_data(lat=lat, lon=lon, region=region)
    return build_standard_record(
        country_code=country,
        region=region,
        lat=lat,
        lon=lon,
        soil_data=soil,
        weather_data=weather,
        satellite_data=satellite,
        advisory_data=advisory_data
    )


@app.post("/api/interop/validate")
def validate_agridata(payload: Dict[str, Any]):
    """Validates an incoming foreign agricultural data payload against the AgriData schema."""
    return validate_agridata_payload(payload)


@app.get("/api/interop/sample-nodes")
def get_sample_interop_nodes():
    """Returns sample cross-border agricultural nodes participating in the network."""
    return get_sample_crossborder_records()


# ---------- 8. FARMS & PERSISTENCE ----------
@app.get("/api/farms")
def get_farms_endpoint():
    """Lists saved farms in the platform database."""
    return list_farms()


@app.post("/api/farms")
def create_farm_endpoint(farm: FarmCreateRequest):
    """Registers or updates a farm profile."""
    return save_farm(farm.dict())


@app.get("/api/history")
def get_history_endpoint(limit: int = Query(10, ge=1, le=50)):
    """Retrieves recent advisories and disease scans."""
    return get_recent_history(limit=limit)


# ---------- 9. LOCALIZATION ----------
@app.get("/api/locales")
def get_locales(lang: str = Query("en")):
    """Returns localized UI strings and crop dictionaries for English, Telugu, and Hindi."""
    return get_translation_bundle(lang=lang)
