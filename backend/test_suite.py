"""
AgriLink Comprehensive End-to-End Test Suite
Tests all 13 core platform endpoints and services.
"""

import os
import io
import sys
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.advisory import GOOGLE_CLOUD_GEMINI_MODEL, generate_ai_chat_response

client = TestClient(app)

def run_tests():
    print("==================================================")
    print("   AGRILINK PLATFORM — END-TO-END TEST SUITE     ")
    print("==================================================")
    passed = 0
    total = 0

    def assert_test(name, condition, extra=""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name} {extra}")
        else:
            print(f"  [FAIL] {name} {extra}")

    # 1. Health
    r = client.get("/health")
    assert_test("Health Endpoint (/health)", r.status_code == 200 and r.json().get("status") == "healthy")

    # Unified Cloud Run deployment serves the frontend and its assets from FastAPI.
    r = client.get("/")
    assert_test("Frontend served at service root", r.status_code == 200 and "text/html" in r.headers.get("content-type", "") and "window.location.origin" in r.text)
    r = client.get("/assets/sample_healthy_rice.jpg")
    assert_test("Frontend demo assets served by backend", r.status_code == 200 and r.headers.get("content-type") == "image/jpeg")

    # 2. Weather
    r = client.get("/api/weather?lat=17.3850&lon=78.4867")
    assert_test("Weather Endpoint (/api/weather)", r.status_code == 200 and "current" in r.json() and "forecast" in r.json(), f"(Source: {r.json().get('source')})")

    # 3. Soil
    r = client.get("/api/soil?region=Telangana")
    assert_test("Soil Profile Endpoint (/api/soil)", r.status_code == 200 and r.json().get("ph") == 7.2 and "status" in r.json())

    # 4. Satellite
    r = client.get("/api/satellite?lat=17.3850&lon=78.4867&region=Telangana")
    assert_test("Satellite Endpoint (/api/satellite)", r.status_code == 200 and "ndvi" in r.json() and r.json().get("data_quality") == "demo")

    # 5. Crop Recommendation
    crop_payload = {
        "nitrogen": 80,
        "phosphorus": 45,
        "potassium": 50,
        "temperature": 27.0,
        "humidity": 75.0,
        "ph": 6.8,
        "rainfall": 200.0,
        "soil_moisture": 0.42,
        "farm_id": "farm-test-01",
        "language": "en"
    }
    r = client.post("/api/crop-recommendation", json=crop_payload)
    data = r.json()
    assert_test("Crop Recommendation ML (/api/crop-recommendation)", r.status_code == 200 and "recommended_crop" in data and len(data.get("reasons", [])) > 0, f"(Crop: {data.get('recommended_crop')}, Conf: {data.get('confidence')})")

    # 6. Disease Detection
    sample_img_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "assets", "sample_rice_blast.jpg")
    if os.path.exists(sample_img_path):
        with open(sample_img_path, "rb") as f:
            img_bytes = f.read()
        files = {"file": ("sample_rice_blast.jpg", img_bytes, "image/jpeg")}
        r = client.post("/api/disease?farm_id=farm-test-01", files=files)
        data = r.json()
        assert_test("Crop Disease Diagnostic (/api/disease)", r.status_code == 200 and "diagnosis" in data and data.get("confidence", 0) > 0.6, f"(Diag: {data.get('diagnosis')})")
    else:
        assert_test("Crop Disease Diagnostic (/api/disease)", False, "Sample image not found")

    # 7. AI Advisory
    adv_payload = {
        "farm_info": {"name": "Test Farm", "region": "Telangana", "latitude": 17.385, "longitude": 78.486, "area_acres": 4.5},
        "weather_data": {"current": {"temperature": 28.0, "humidity": 70.0, "condition": "Mainly clear"}},
        "soil_data": {"ph": 6.8, "nitrogen": 80, "phosphorus": 45, "potassium": 50, "moisture": 0.40, "soil_type": "Black Cotton Soil"},
        "satellite_data": {"ndvi": 0.68, "soil_moisture": 0.40, "land_surface_temp": 32.0, "vegetation_health": "Good"},
        "crop_recommendation": {"recommended_crop": "rice", "confidence": 0.88, "reasons": ["Optimal pH", "Adequate moisture"], "regenerative_tip": "AWD irrigation"},
        "language": "en"
    }
    r = client.post("/api/ai-advisory", json=adv_payload)
    data = r.json()
    assert_test("AI Agricultural Advisor (/api/ai-advisory)", r.status_code == 200 and "field_condition_summary" in data and "water_management" in data, f"(Engine: {data.get('engine')})")

    chat_payload = {
        "question": "What should I prioritize before tomorrow's rain?",
        "history": [{"role": "user", "content": "How is my soil?"}, {"role": "assistant", "content": "Soil pH is 6.8."}],
        "farm_info": {"name": "Test Farm", "region": "Telangana", "lat": 17.385, "lon": 78.486, "area": 4.5, "soilType": "Black Cotton Soil", "currentCrop": "Rice"},
        "weather_data": {"current": {"temperature": 29.5, "humidity": 70, "precipitation": 0.0, "condition": "Cloudy"}, "forecast": [{"date": "tomorrow", "precipitation": 12.0}]},
        "soil_data": {"ph": 6.8, "nitrogen": 80, "phosphorus": 45, "potassium": 50, "moisture": 0.4, "soilType": "Black Cotton Soil"},
        "satellite_data": {"ndvi": 0.68, "soil_moisture": 0.4, "land_surface_temp": 32.0, "vegetation_health": "Good"},
        "crop_recommendation": {"recommended_crop": "rice", "confidence": 0.88},
        "language": "en"
    }
    with patch("backend.main.generate_ai_chat_response", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "Clear the drainage channels before the forecast rain."
        r = client.post("/api/ai-chat", json=chat_payload)
    chat_data = r.json()
    chat_context = mock_chat.await_args.kwargs["context"]
    assert_test("Vertex AI chat answers the submitted question", r.status_code == 200 and chat_data.get("answer") == mock_chat.return_value)
    assert_test("Vertex AI chat includes normalized farm and forecast context", chat_context["farm"]["coordinates"]["lat"] == 17.385 and chat_context["weather_forecast"][0]["precipitation"] == 12.0 and len(mock_chat.await_args.kwargs["history"]) == 2)

    mock_generate = AsyncMock(return_value=SimpleNamespace(text="Drain the field before the rain."))
    fake_client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=mock_generate)))
    with patch("backend.services.advisory._get_vertex_client", return_value=fake_client), patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-project"}):
        answer = asyncio.run(generate_ai_chat_response(
            question="How should I prepare for tomorrow's rain?",
            context={"weather_forecast": [{"precipitation": 12}]},
            history=[{"role": "user", "content": "What is my soil type?"}, {"role": "assistant", "content": "Black cotton soil."}]
        ))
    request = mock_generate.await_args.kwargs
    assert_test("Vertex AI uses Flash-Lite with minimal thinking", answer == "Drain the field before the rain." and request["model"] == GOOGLE_CLOUD_GEMINI_MODEL and request["config"].thinking_config.thinking_level == "MINIMAL")

    # 8. Interoperability Schema
    r = client.get("/api/schema")
    assert_test("Interoperability Schema (/api/schema)", r.status_code == 200 and r.json().get("version") == "1.0.0")

    # 9. Interoperability Export
    r = client.post("/api/interop/export?country=IND&region=Telangana&lat=17.385&lon=78.486")
    assert_test("AgriData Export (/api/interop/export)", r.status_code == 200 and "metadata" in r.json() and "observations" in r.json())

    # 10. Interoperability Validate
    sample_record = r.json()
    r = client.post("/api/interop/validate", json=sample_record)
    assert_test("AgriData Validation (/api/interop/validate)", r.status_code == 200 and r.json().get("valid") is True)

    # 11. Farms Endpoint
    r = client.get("/api/farms")
    assert_test("Farms List Endpoint (/api/farms)", r.status_code == 200 and len(r.json()) >= 1)

    # 12. History Endpoint
    r = client.get("/api/history?limit=5")
    assert_test("History Persistence Endpoint (/api/history)", r.status_code == 200 and "advisories" in r.json() and "disease_scans" in r.json())

    # 13. Localization
    r = client.get("/api/locales?lang=te")
    assert_test("Localization Telugu (/api/locales?lang=te)", r.status_code == 200 and r.json().get("lang") == "te")
    r = client.get("/api/locales?lang=hi")
    assert_test("Localization Hindi (/api/locales?lang=hi)", r.status_code == 200 and r.json().get("lang") == "hi")

    print("==================================================")
    print(f"   TEST SUMMARY: {passed}/{total} Passed ({(passed/total)*100:.1f}%)")
    print("==================================================")

    if passed == total:
        print(">>> SUCCESS: ALL TESTS PASSED! System is fully operational and demo-ready.")
        return 0
    else:
        print(">>> WARNING: Some tests failed. Check logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
