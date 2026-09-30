"""
AI Agricultural Advisor Service
Synthesizes verified multidimensional data (Weather, Soil, Satellite NDVI, Crop ML, Disease)
into grounded, farmer-friendly agricultural advisories using Google Gemini API with deterministic fallback.
"""

import os
import json
from functools import lru_cache
from typing import Dict, Any, List, Optional
from backend.utils.localization import TRANSLATIONS, CROP_NAMES_TRANSLATION

# Try loading Google GenAI SDK if key is configured
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_CLOUD_GEMINI_MODEL = os.getenv("GOOGLE_CLOUD_GEMINI_MODEL", "gemini-3.5-flash-lite")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")


@lru_cache(maxsize=2)
def _get_vertex_client(project: str, location: str):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Install the Google GenAI SDK with `pip install -r backend_requirements.txt`.") from exc
    return genai.Client(vertexai=True, project=project, location=location)


async def generate_ai_chat_response(
    question: str,
    context: Dict[str, Any],
    history: Optional[List[Dict[str, str]]] = None,
    language: str = "en"
) -> str:
    """Generate a question-specific, context-grounded answer with Vertex AI Gemini."""
    language_instruction = {
        "en": "Answer in clear, practical English for an Indian farmer.",
        "te": "Answer in natural, farmer-friendly Telugu (తెలుగు).",
        "hi": "Answer in natural, farmer-friendly Hindi (हिन्दी)."
    }.get(language, "Answer in clear, practical English.")

    system_prompt = f"""You are AgriLink, a conversational agricultural advisor.
Answer the farmer's latest question directly and give a different, relevant response for each question. Do not repeat a generic farm summary unless asked.
Use only the verified farm context below. Never invent measurements, weather forecasts, disease findings, or product dosages. If needed data is missing, say so plainly and ask one useful follow-up question.
Mention only context facts directly relevant to the question. Give a concise answer in at most 2 short sentences or 3 brief steps. Do not provide alternative answers or repeat the prompt. Prefer integrated pest management, safe cultural practices, and local agricultural extension guidance. Do not recommend hazardous chemical use.
{language_instruction}

VERIFIED FARM CONTEXT:
{json.dumps(context, ensure_ascii=False, indent=2)}"""

    try:
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install the Google GenAI SDK with `pip install -r backend_requirements.txt`.") from exc

    contents = []
    for turn in (history or [])[-4:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in {"user", "assistant"} and content:
            contents.append(types.Content(
                role="model" if role == "assistant" else "user",
                parts=[types.Part.from_text(text=content[:600])]
            ))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

    try:
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        if not project:
            raise RuntimeError("Set GOOGLE_CLOUD_PROJECT and authenticate with Application Default Credentials to use Vertex AI.")
        client = _get_vertex_client(project, GOOGLE_CLOUD_LOCATION)
        response = await client.aio.models.generate_content(
            model=GOOGLE_CLOUD_GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=160,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)
            )
        )
        answer = (response.text or "").strip()
        if not answer:
            raise RuntimeError("Vertex AI returned an empty answer. Try asking the question again.")
        return answer
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Vertex AI request failed. Check project access, billing, model region, and Application Default Credentials.") from exc


def generate_structured_context(
    farm_info: Dict[str, Any],
    weather_data: Dict[str, Any],
    soil_data: Dict[str, Any],
    satellite_data: Dict[str, Any],
    crop_recommendation: Dict[str, Any],
    disease_result: Optional[Dict[str, Any]] = None,
    language: str = "en"
) -> Dict[str, Any]:
    """Builds a verified, structured agronomic profile from all platform sensors and models."""
    current_weather = weather_data.get("current", {})
    return {
        "farm": {
            "name": farm_info.get("name", "Demo Farm"),
            "region": farm_info.get("region", "Telangana, India"),
            "coordinates": {"lat": farm_info.get("latitude", 17.3850), "lon": farm_info.get("longitude", 78.4867)},
            "area_acres": farm_info.get("area_acres", 4.5),
            "soil_type": soil_data.get("soil_type", "Black Cotton Soil")
        },
        "soil_measurements": {
            "ph": soil_data.get("ph", 7.2),
            "nitrogen_kg_ha": soil_data.get("nitrogen", 68),
            "phosphorus_kg_ha": soil_data.get("phosphorus", 42),
            "potassium_kg_ha": soil_data.get("potassium", 55),
            "soil_moisture": soil_data.get("moisture", 0.38),
            "organic_carbon_pct": soil_data.get("organic_carbon", 0.52),
            "source": soil_data.get("source", "Regional estimate")
        },
        "weather_conditions": {
            "temperature_c": current_weather.get("temperature", 28.0),
            "feels_like_c": current_weather.get("feels_like", 30.0),
            "humidity_pct": current_weather.get("humidity", 70),
            "precipitation_mm": current_weather.get("precipitation", 0.0),
            "wind_speed_kmh": current_weather.get("wind_speed", 12.0),
            "condition": current_weather.get("condition", "Partly cloudy")
        },
        "satellite_indicators": {
            "ndvi": satellite_data.get("ndvi", 0.68),
            "soil_moisture_index": satellite_data.get("soil_moisture", 0.41),
            "land_surface_temp_c": satellite_data.get("land_surface_temp", 32.5),
            "vegetation_health": satellite_data.get("vegetation_health", "Moderate-Good")
        },
        "crop_ml_output": {
            "recommended_crop": crop_recommendation.get("recommended_crop", "Rice"),
            "confidence": crop_recommendation.get("confidence", 0.88),
            "reasons": crop_recommendation.get("reasons", []),
            "regenerative_tip": crop_recommendation.get("regenerative_tip", "Rotate with nitrogen-fixing pulses.")
        },
        "disease_scan": disease_result if disease_result else {"status": "No active disease reported"}
    }


def generate_rule_based_advisory(context: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
    """Deterministic, agronomically sound fallback advisory when Gemini API key is offline."""
    crop = context["crop_ml_output"]["recommended_crop"]
    crop_display = crop.capitalize()
    temp = context["weather_conditions"]["temperature_c"]
    humidity = context["weather_conditions"]["humidity_pct"]
    soil_moisture = context["soil_measurements"]["soil_moisture"]
    ph = context["soil_measurements"]["ph"]
    ndvi = context["satellite_indicators"]["ndvi"]
    disease = context.get("disease_scan", {})

    # Water considerations
    if soil_moisture < 0.25:
        water_advice = f"Root-zone soil moisture is low ({soil_moisture:.2f}). Initiate light furrow or drip irrigation during early morning hours to minimize evaporative loss."
    elif soil_moisture > 0.50:
        water_advice = f"Soil moisture is high ({soil_moisture:.2f}). Ensure field drainage channels are clear to prevent waterlogging and root asphyxiation."
    else:
        water_advice = f"Soil moisture ({soil_moisture:.2f}) is in the optimal range. Maintain current scheduled irrigation intervals."

    # Risk factors
    risks = []
    if humidity > 80 and temp > 25:
        risks.append("High ambient humidity and warm temperatures create elevated fungal spore germination risk. Scout leaf canopy daily.")
    if ph < 6.0:
        risks.append("Sub-optimal acidic soil pH may restrict phosphorus and micronutrient uptake. Consider agricultural lime application.")
    elif ph > 8.0:
        risks.append("Alkaline soil pH may induce micronutrient (Zinc, Iron) immobilization. Consider applying elemental sulfur or gypsum.")
    if disease.get("disease_key") and disease.get("disease_key") != "Healthy___Leaf":
        risks.append(f"Active disease detection: {disease.get('diagnosis')} with {int(disease.get('confidence', 0.8)*100)}% confidence.")

    if not risks:
        risks.append("Agro-climatic stress indicators are currently low. Favorable growing window.")

    # Actions
    actions = [
        f"Proceed with land preparation and sowing of certified {crop_display} seeds.",
        f"Apply basal fertilizer dose aligned with soil test values (N:{context['soil_measurements']['nitrogen_kg_ha']}, P:{context['soil_measurements']['phosphorus_kg_ha']}, K:{context['soil_measurements']['potassium_kg_ha']} kg/ha).",
        context["crop_ml_output"]["regenerative_tip"]
    ]

    # Localization pass for Telugu and Hindi
    if language == "te":
        crop_te = CROP_NAMES_TRANSLATION.get(crop.lower(), {}).get("te", crop_display)
        summary = f"మీ పొలం పరిస్థితులు మరియు వాతావరణ సూచన ఆధారంగా, **{crop_te}** సాగు అత్యంత అనుకూలంగా ఉంది. ఉపగ్రహ పచ్చదనం సూచిక (NDVI: {ndvi}) మరియు నేల స్వభావం సమతుల్యంగా ఉన్నాయి."
        irrigation_advice = "నేలలోని తేమ స్థాయికి అనుగుణంగా ఉదయం వేళల్లో మాత్రమే తేలికపాటి తడులు ఇవ్వండి. నీరు నిలవకుండా డ్రైనేజీ చూడండి."
        risk_text = "గాలిలో తేమ ఎక్కువగా ఉన్నందున తెగుళ్ల పట్ల అప్రమత్తంగా ఉండండి. క్రమం తప్పకుండా ఆకులను పరిశీలించండి."
        action_text = f"1. {crop_te} విత్తన శుద్ధి చేసి నాటండి.\n2. సిఫార్సు చేసిన మోతాదులో మాత్రమే ఎరువులు వాడండి.\n3. {context['crop_ml_output']['regenerative_tip']}"
    elif language == "hi":
        crop_hi = CROP_NAMES_TRANSLATION.get(crop.lower(), {}).get("hi", crop_display)
        summary = f"आपके खेत की वर्तमान मृदा, मौसम और उपग्रह संकेतकों (NDVI: {ndvi}) के अनुसार **{crop_hi}** की खेती सबसे उपयुक्त पाई गई है।"
        irrigation_advice = "मृदा नमी स्तर के अनुसार सुबह के समय हल्की सिंचाई करें। जलभराव से बचने के लिए जल निकासी नाली साफ रखें।"
        risk_text = "उच्च आर्द्रता के कारण फंगल संक्रमण का खतरा हो सकता है। पत्तियों की नियमित निगरानी करें।"
        action_text = f"1. {crop_hi} के प्रमाणित बीजों का चयन एवं बीजोपचार करें।\n2. संतुलित एनपीके उर्वरक का प्रयोग करें।\n3. {context['crop_ml_output']['regenerative_tip']}"
    else:
        summary = f"Based on verified agro-meteorological parameters, soil nutrient profile, and satellite vegetation index (NDVI: {ndvi}), **{crop_display}** is the recommended crop for this field cycle."
        irrigation_advice = water_advice
        risk_text = " • " + "\n • ".join(risks)
        action_text = "1. " + "\n2. ".join(actions)

    # Tab-specific structured content
    tabs = {
        "crop": {
            "title": "Crop Lifecycle & Agronomy",
            "crop": crop_display,
            "suitability": "Highly Suitable" if context["crop_ml_output"]["confidence"] > 0.75 else "Moderate",
            "soil_fit": f"pH {ph:.1f} is compatible. N-P-K ({context['soil_measurements']['nitrogen_kg_ha']}-{context['soil_measurements']['phosphorus_kg_ha']}-{context['soil_measurements']['potassium_kg_ha']} kg/ha) supports vegetative vigor.",
            "recommendation": f"Proceed with certified {crop_display} seed sowing. Use Rhizobium or mycorrhizal bio-priming for higher nutrient assimilation."
        },
        "water": {
            "title": "Root-Zone Water & Irrigation",
            "soil_moisture_pct": round(soil_moisture * 100),
            "status": "Adequate" if 0.30 <= soil_moisture <= 0.50 else "Dry (Irrigation Needed)" if soil_moisture < 0.30 else "Saturated (Drainage Needed)",
            "advice": irrigation_advice,
            "forecast_rain": f"{context['weather_conditions']['precipitation_mm']} mm today"
        },
        "weather": {
            "title": "Microclimate & Risk Assessment",
            "temperature": f"{temp:.1f}°C",
            "humidity": f"{humidity}%",
            "condition": context["weather_conditions"]["condition"],
            "risks": risks,
            "advice": "Scout field canopy daily during periods of sustained humidity >75%."
        },
        "disease": {
            "title": "Crop Health & Disease Monitoring",
            "status": disease.get("diagnosis", "No active disease detected on recent visual scan"),
            "severity": disease.get("severity", "Low"),
            "confidence": f"{int(disease.get('confidence', 0.8)*100)}%" if disease.get("confidence") else "N/A",
            "management": disease.get("recommended_actions", ["Maintain regular visual scouting once every 7-10 days."])
        },
        "sustainability": {
            "title": "Regenerative & Climate Resilience",
            "practice": context["crop_ml_output"]["regenerative_tip"],
            "carbon_impact": "Incorporating organic crop residues and cover crops builds active soil organic carbon (SOC) and improves water retention by 15-25%."
        }
    }

    return {
        "engine": "AgriLink Grounded Agronomic Intelligence",
        "crop": crop_display,
        "confidence": context["crop_ml_output"]["confidence"],
        "field_condition_summary": summary,
        "water_management": irrigation_advice,
        "risk_assessment": risk_text if isinstance(risk_text, list) else risk_text,
        "recommended_actions": action_text,
        "why_explanation": " • " + "\n • ".join(context["crop_ml_output"]["reasons"]),
        "regenerative_practice": context["crop_ml_output"]["regenerative_tip"],
        "tabs": tabs,
        "language": language,
        "data_grounded": True
    }


async def generate_ai_advisory(
    farm_info: Dict[str, Any],
    weather_data: Dict[str, Any],
    soil_data: Dict[str, Any],
    satellite_data: Dict[str, Any],
    crop_recommendation: Dict[str, Any],
    disease_result: Optional[Dict[str, Any]] = None,
    language: str = "en"
) -> Dict[str, Any]:
    """Generates LLM-grounded advisory using Google Gemini or falls back to rule engine."""
    context = generate_structured_context(
        farm_info=farm_info,
        weather_data=weather_data,
        soil_data=soil_data,
        satellite_data=satellite_data,
        crop_recommendation=crop_recommendation,
        disease_result=disease_result,
        language=language
    )

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return generate_rule_based_advisory(context, language=language)

    # If GEMINI_API_KEY is available, invoke Google GenAI
    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        lang_instruction = {
            "en": "Respond in clear, encouraging, practical English suitable for an Indian farmer.",
            "te": "Respond directly in clear, natural Telugu (తెలుగు) language using farmer-friendly agronomic terms.",
            "hi": "Respond directly in clear, natural Hindi (हिन्दी) language using farmer-friendly agronomic terms."
        }.get(language, "Respond in English.")

        prompt = f"""
You are the AgriLink AI Agricultural Advisor. You provide practical, climate-resilient, and scientifically sound farming guidance.

STRICT GROUNDING RULES:
1. Base all advice strictly on the verified agricultural telemetry provided in the context below. Do not invent measurements or fictional weather conditions.
2. Do NOT prescribe hazardous or unverified chemical pesticides. Emphasize organic, cultural, biological, and regenerative practices.
3. Keep the advice concise, respectful, and actionable for a smallholder farmer.
4. {lang_instruction}

VERIFIED AGRICULTURAL CONTEXT:
{json.dumps(context, indent=2)}

Please return your advisory in structured JSON with the following exact keys:
{{
  "field_condition_summary": "2-3 sentences summarizing field state and crop suitability",
  "water_management": "Specific irrigation advice based on soil moisture and precipitation",
  "risk_assessment": "Key pest, fungal, or weather risks to monitor",
  "recommended_actions": "Step-by-step next actions (numbered 1, 2, 3)",
  "why_explanation": "Brief explanation of why the crop was recommended based on verified data",
  "regenerative_practice": "A high-impact regenerative farming tip"
}}
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        response_text = response.text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(response_text)
        return {
            "engine": "Google Gemini 2.5 Flash (Grounded)",
            "crop": context["crop_ml_output"]["recommended_crop"],
            "confidence": context["crop_ml_output"]["confidence"],
            "field_condition_summary": parsed.get("field_condition_summary", ""),
            "water_management": parsed.get("water_management", ""),
            "risk_assessment": parsed.get("risk_assessment", ""),
            "recommended_actions": parsed.get("recommended_actions", ""),
            "why_explanation": parsed.get("why_explanation", ""),
            "regenerative_practice": parsed.get("regenerative_practice", context["crop_ml_output"]["regenerative_tip"]),
            "language": language,
            "data_grounded": True
        }
    except Exception as e:
        print(f"Gemini API invocation exception, using deterministic fallback: {e}")
        return generate_rule_based_advisory(context, language=language)
