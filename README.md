# AgriLink — AI-Powered Interoperable Digital Agriculture Platform

AgriLink is an open, standards-based digital agriculture intelligence platform built for climate resilience and cross-border agronomic interoperability (inspired by BRICS AgriN and UN FAO Digital Public Goods).

---

## 🌟 Key Platform Capabilities

1. **Live Agro-Meteorological Telemetry**: Real-time Open-Meteo integration delivering temperature, humidity, precipitation, wind speed, and 7-day weather forecasts.
2. **Soil Chemical & Physical Diagnostics**: Regional soil profile baselines (pH, Nitrogen, Phosphorus, Potassium, Moisture, Organic Carbon) with threshold status chips and manual soil test overrides.
3. **Satellite Remote Sensing (NDVI / Moisture / LST)**: Vegetation health indicators, Land Surface Temperature, and root-zone moisture metrics with clear provenance tagging.
4. **ML Crop Recommendation with Explainability**: 15-crop tabular RandomForestClassifier providing calibrated confidence scores and multi-factor agronomic factor checks ("Why this crop?").
5. **Vision-Based Crop Disease Diagnostic**: Leaf visual analysis detecting pathogens (Rice Blast, Brown Spot, Early/Late Blight, Rust, Bacterial Blight) with calibrated confidence and non-chemical agronomic guidance.
6. **Grounded AI Agricultural Advisor**: Conversational farm guidance generated with Google Cloud Vertex AI Gemini from the farmer's question, recent chat, and verified soil, weather, crop, satellite, and disease data. Structured farm analysis retains its existing Gemini/rules fallback.
7. **Cross-Border Interoperability Standard (AgriData v1.0.0)**: Standardized schema contract for data exchange between national agricultural nodes (e.g. India ⇄ Brazil ⇄ Kenya).
8. **Multi-Language Localization**: Full UI and advisory translation in **English**, **Telugu (తెలుగు)**, and **Hindi (हिन्दी)**.
9. **Interactive Field Map**: Leaflet.js mapping with GPS coordinates and farm boundary visualization.
10. **Zero-Config Database Persistence**: SQLite tracking farms, weather observations, crop recommendations, and disease scan audits.

---

## 🚀 Quickstart & Running the Platform

### 1. Backend Setup & Startup
```bash
# Optional: create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install -r backend_requirements.txt

# Train Crop Model (Pre-trained bundle included in backend/models/crop_model.pkl)
python backend/train_model.py

# Launch FastAPI Server on port 8000
uvicorn backend.main:app --reload --port 8000
```
Interactive OpenAPI Documentation: `http://localhost:8000/docs`

### 2. Google Cloud Vertex AI Advisor
Enable billing and the Vertex AI API for your Google Cloud project, then install the Google Cloud CLI and authenticate Application Default Credentials:
```powershell
gcloud auth application-default login
```
Set these values in the same PowerShell session used to launch FastAPI:
```powershell
$env:GOOGLE_CLOUD_PROJECT = "your-google-cloud-project-id"
$env:GOOGLE_CLOUD_LOCATION = "global"
$env:GOOGLE_CLOUD_GEMINI_MODEL = "gemini-3.5-flash-lite"
python -m pip install -r backend_requirements.txt
uvicorn backend.main:app --reload --port 8000
```
The dashboard chat sends each question to `/api/ai-chat` using Vertex AI and Application Default Credentials; no service-account key or API key is stored in the repository. The existing complete farm analysis endpoint is unchanged. The chat uses Gemini Flash-Lite's minimal thinking setting for lower latency.

### 3. Launching Frontend Dashboard
Open `frontend/index.html` (or `frontend_index.html`) in any modern browser.

---

## 🧪 Running Automated Test Suite
```bash
python backend/test_suite.py
```
Validates all 14 endpoints including weather, soil, ML inference, disease vision diagnosis, AI advisory, interoperability schema, and database persistence.

---

## 🧭 Hackathon Judge Demo Flow (1-Minute Walkthrough)

1. **Select Region**: Choose **Telangana (Hyderabad)** from the top-left dropdown. The map pans, live Open-Meteo weather loads, and regional soil metrics populate.
2. **One-Click Analysis**: Click **"🚀 Analyze Farm & Advise"**.
3. **Inspect Explainability**: Under *Crop Recommendation*, see the recommended crop (e.g., **Rice (Paddy)**), calibrated confidence bar (89%), and the factor checklist explaining *why* (pH, thermal requirements, moisture).
4. **Test Disease Diagnostic**: Under *Disease Diagnostic*, click one of the quick test buttons (e.g. **"Rice Blast"**). The vision classifier instantly analyzes lesion patterns, displays severity, and provides safe cultural remedies.
5. **Switch Language**: Change the language dropdown to **తెలుగు (Telugu)** or **हिन्दी (Hindi)** to see the entire UI and AI Advisory adapt instantaneously.
6. **Demonstrate Interoperability**: In the *AgriData Standard* card, click **"Exchange Demo"** to see live cross-border data records between India, Brazil, and Kenya.
7. **View Audit History**: Scroll down to see all generated advisories and disease scans automatically recorded in SQLite.

---

## 📐 Standardized AgriData Interoperability Architecture

```
[ Regional Sensor & Weather IoT ] ───┐
[ Soil Survey & Farmer Inputs  ] ───┼──> [ AgriLink Standardization Node ] ──> [ Common AgriData Schema (JSON) ]
[ Satellite Remote Sensing     ] ───┘                     │
                                                          ├──> [ Machine Learning & Gemini Advisor ]
                                                          └──> [ Cross-Border Exchange Gateway ]
```

---

## 🛡️ Security & Privacy
- No hardcoded API secrets.
- Environment variables configured via `.env.example`.
- Clean error boundaries and fallback routines preventing raw stack traces.
