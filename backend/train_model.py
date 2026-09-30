"""
AgriLink Crop Recommendation Model Training
Trains a multi-crop RandomForestClassifier on agronomic parameters:
N, P, K, Temperature, Humidity, Soil pH, Rainfall, Soil Moisture.
Includes rule-based agronomic explainability engine.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Agronomic optimal ranges for 15 primary crops across Indian and global agro-climatic zones
CROP_AGRONOMIC_RANGES = {
    "rice": {
        "N": (80, 130), "P": (40, 65), "K": (40, 65),
        "temp": (22, 34), "hum": (70, 95), "ph": (5.5, 7.2), "rain": (180, 320), "moisture": (0.40, 0.65),
        "regen_tip": "Adopt Alternate Wetting and Drying (AWD) irrigation to reduce methane emissions by 30-50% and retain organic straw."
    },
    "wheat": {
        "N": (90, 140), "P": (45, 70), "K": (35, 60),
        "temp": (12, 26), "hum": (45, 70), "ph": (6.0, 7.8), "rain": (50, 120), "moisture": (0.30, 0.45),
        "regen_tip": "Practice zero-tillage direct drilling into happy seeder rice residue; intercrop with chickpea to naturally replenish soil N."
    },
    "maize": {
        "N": (80, 120), "P": (40, 60), "K": (30, 50),
        "temp": (18, 32), "hum": (50, 80), "ph": (5.8, 7.5), "rain": (70, 160), "moisture": (0.32, 0.48),
        "regen_tip": "Intercrop with legume cover crops (cowpea/pigeonpea) and retain post-harvest stalks as mulch to build organic carbon."
    },
    "cotton": {
        "N": (90, 140), "P": (30, 55), "K": (30, 55),
        "temp": (22, 36), "hum": (45, 75), "ph": (6.0, 8.2), "rain": (60, 130), "moisture": (0.28, 0.42),
        "regen_tip": "Integrate high-density planting with drip fertigation and border crops of marigold/castor to disrupt bollworm cycles."
    },
    "sugarcane": {
        "N": (140, 220), "P": (60, 95), "K": (80, 130),
        "temp": (24, 38), "hum": (65, 90), "ph": (6.0, 7.8), "rain": (140, 260), "moisture": (0.42, 0.60),
        "regen_tip": "Practice trash mulching between cane rows to cut evaporation losses and split N application to prevent nitrate leaching."
    },
    "chickpea": {
        "N": (15, 35), "P": (40, 65), "K": (20, 35),
        "temp": (14, 28), "hum": (35, 60), "ph": (6.0, 7.8), "rain": (35, 85), "moisture": (0.22, 0.36),
        "regen_tip": "Inoculate seeds with Rhizobium & PSB culture; maintain minimal tillage to preserve mycorrhizal fungi networks."
    },
    "groundnut": {
        "N": (15, 35), "P": (35, 60), "K": (35, 60),
        "temp": (22, 33), "hum": (50, 75), "ph": (5.6, 7.2), "rain": (55, 110), "moisture": (0.26, 0.40),
        "regen_tip": "Apply gypsum (200 kg/ha) at pegging stage and incorporate shell waste into compost to enrich calcium and sulfur."
    },
    "mustard": {
        "N": (60, 95), "P": (30, 50), "K": (20, 40),
        "temp": (10, 24), "hum": (40, 65), "ph": (6.0, 7.5), "rain": (30, 70), "moisture": (0.24, 0.38),
        "regen_tip": "Use wider line spacing for natural ventilation against Alternaria blight and apply bio-sulfur for oil yield."
    },
    "sorghum": {
        "N": (60, 100), "P": (30, 50), "K": (25, 45),
        "temp": (22, 35), "hum": (40, 65), "ph": (6.0, 8.0), "rain": (40, 90), "moisture": (0.22, 0.35),
        "regen_tip": "Ideal climate-resilient dryland crop; use contour bunding to harvest rainwater and rotate with pulse crops."
    },
    "pearl_millet": {
        "N": (50, 80), "P": (25, 45), "K": (20, 40),
        "temp": (25, 38), "hum": (30, 60), "ph": (6.2, 8.4), "rain": (30, 70), "moisture": (0.20, 0.32),
        "regen_tip": "Highly drought-tolerant millet; integrate with agroforestry (Khejri/Neem) for microclimate moderation."
    },
    "soybean": {
        "N": (20, 45), "P": (50, 80), "K": (35, 60),
        "temp": (20, 32), "hum": (55, 80), "ph": (6.0, 7.2), "rain": (80, 150), "moisture": (0.32, 0.48),
        "regen_tip": "Broadbed and furrow (BBF) planting to manage monsoon excess moisture while fixing up to 100 kg N/ha in soil."
    },
    "turmeric": {
        "N": (90, 130), "P": (45, 70), "K": (70, 110),
        "temp": (20, 34), "hum": (65, 85), "ph": (5.5, 7.5), "rain": (120, 220), "moisture": (0.38, 0.55),
        "regen_tip": "Heavy organic mulching with green leaves (15 t/ha) at planting suppresses weeds and builds long-term humus."
    },
    "chilli": {
        "N": (80, 120), "P": (40, 65), "K": (50, 80),
        "temp": (20, 33), "hum": (55, 75), "ph": (6.0, 7.2), "rain": (60, 120), "moisture": (0.28, 0.42),
        "regen_tip": "Use silver-black reflective plastic or straw mulch to reduce sucking pest incidence and conserve root-zone water."
    },
    "pigeonpea": {
        "N": (20, 40), "P": (40, 65), "K": (25, 45),
        "temp": (22, 35), "hum": (45, 70), "ph": (6.2, 7.8), "rain": (55, 110), "moisture": (0.25, 0.38),
        "regen_tip": "Deep taproot system opens subsoil plow pans; excellent as intercrop in cotton and sorghum."
    },
    "coffee": {
        "N": (70, 110), "P": (30, 50), "K": (70, 110),
        "temp": (16, 26), "hum": (70, 90), "ph": (5.0, 6.5), "rain": (150, 280), "moisture": (0.38, 0.55),
        "regen_tip": "Maintain diverse multi-tier shade canopy with native trees to foster pollinators and natural pest predators."
    }
}

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_moisture"]


def sample_crop_data(crop_name: str, ranges: dict, n_samples: int = 350) -> list:
    samples = []
    for _ in range(n_samples):
        # Sample with slight Gaussian noise around uniform ranges for robust variance
        row = {
            "N": max(5, np.random.uniform(*ranges["N"]) + np.random.normal(0, 3)),
            "P": max(5, np.random.uniform(*ranges["P"]) + np.random.normal(0, 2)),
            "K": max(5, np.random.uniform(*ranges["K"]) + np.random.normal(0, 2)),
            "temperature": np.random.uniform(*ranges["temp"]) + np.random.normal(0, 0.8),
            "humidity": np.clip(np.random.uniform(*ranges["hum"]) + np.random.normal(0, 2), 10, 100),
            "ph": np.clip(np.random.uniform(*ranges["ph"]) + np.random.normal(0, 0.1), 4.0, 9.0),
            "rainfall": max(10, np.random.uniform(*ranges["rain"]) + np.random.normal(0, 8)),
            "soil_moisture": np.clip(np.random.uniform(*ranges["moisture"]) + np.random.normal(0, 0.02), 0.1, 0.9),
            "label": crop_name,
        }
        samples.append(row)
    return samples


def generate_explainable_reasons(crop_name: str, input_params: dict) -> list:
    """Compares farm measurements with the agronomic target envelope to give clear explainable reasons."""
    ranges = CROP_AGRONOMIC_RANGES.get(crop_name)
    if not ranges:
        return ["Meets general agro-climatic criteria."]

    reasons = []
    ph = input_params.get("ph", 6.5)
    temp = input_params.get("temperature", 25.0)
    rain = input_params.get("rainfall", 100.0)
    hum = input_params.get("humidity", 60.0)
    n = input_params.get("nitrogen", 70.0)
    p = input_params.get("phosphorus", 40.0)
    k = input_params.get("potassium", 50.0)
    moisture = input_params.get("soil_moisture", 0.35)

    # pH check
    if ranges["ph"][0] - 0.3 <= ph <= ranges["ph"][1] + 0.3:
        reasons.append(f"Soil pH ({ph:.1f}) is within the optimal {ranges['ph'][0]}-{ranges['ph'][1]} range for {crop_name.capitalize()}")
    else:
        reasons.append(f"Soil pH ({ph:.1f}) is tolerable with appropriate conditioning")

    # Temp check
    if ranges["temp"][0] <= temp <= ranges["temp"][1]:
        reasons.append(f"Current temperature ({temp:.1f}°C) matches thermal requirements ({ranges['temp'][0]}-{ranges['temp'][1]}°C)")

    # Rain check
    if rain >= ranges["rain"][0] * 0.8:
        reasons.append(f"Precipitation & water availability ({rain:.0f}mm) supports {crop_name.capitalize()} cultivation")
    else:
        reasons.append(f"Low rainfall ({rain:.0f}mm) requires supplemental micro-irrigation")

    # Soil nutrients
    if n >= ranges["N"][0] * 0.7:
        reasons.append(f"Soil Nitrogen ({n:.0f} kg/ha) is sufficient for initial vegetative vigor")
    if p >= ranges["P"][0] * 0.7 and k >= ranges["K"][0] * 0.7:
        reasons.append(f"Phosphorus ({p:.0f} kg/ha) and Potassium ({k:.0f} kg/ha) levels support healthy root & grain formation")

    if moisture >= ranges["moisture"][0] * 0.8:
        reasons.append(f"Root-zone soil moisture fraction ({moisture:.2f}) provides adequate hydration")

    return reasons[:4]


def get_crop_compatibility_breakdown(crop_name: str, input_params: dict) -> dict:
    """Generates detailed comparison analysis for alternative crops."""
    ranges = CROP_AGRONOMIC_RANGES.get(crop_name.lower())
    if not ranges:
        return {
            "crop": crop_name,
            "compatibility_score": 70,
            "advantages": ["Viable regional crop choice"],
            "limitations": ["Requires localized soil test adjustment"],
            "requirements": "Standard agro-climatic conditions",
            "regen_tip": "Maintain crop rotation."
        }

    ph = input_params.get("ph", 6.8)
    temp = input_params.get("temperature", 28.0)
    rain = input_params.get("rainfall", 150.0)
    n = input_params.get("nitrogen", 70.0)
    moist = input_params.get("soil_moisture", 0.38)

    score = 100
    advantages = []
    limitations = []

    # Check pH
    if ranges["ph"][0] <= ph <= ranges["ph"][1]:
        advantages.append(f"Optimal soil pH ({ph:.1f} vs required {ranges['ph'][0]}-{ranges['ph'][1]})")
    else:
        score -= 15
        limitations.append(f"Soil pH ({ph:.1f}) is slightly outside ideal {ranges['ph'][0]}-{ranges['ph'][1]} (apply soil conditioner)")

    # Check Temp
    if ranges["temp"][0] <= temp <= ranges["temp"][1]:
        advantages.append(f"Ideal thermal range ({temp:.1f}°C fits {ranges['temp'][0]}-{ranges['temp'][1]}°C)")
    else:
        score -= 15
        limitations.append(f"Ambient temperature ({temp:.1f}°C) may stress growth (optimal {ranges['temp'][0]}-{ranges['temp'][1]}°C)")

    # Check Rain
    if rain >= ranges["rain"][0]:
        advantages.append(f"Adequate seasonal rainfall ({rain:.0f}mm vs minimum {ranges['rain'][0]}mm)")
    else:
        score -= 20
        limitations.append(f"Rainfall ({rain:.0f}mm) is below ideal {ranges['rain'][0]}mm — needs drip irrigation")

    if n >= ranges["N"][0] * 0.7:
        advantages.append("Soil nitrogen supports early canopy development")

    if not advantages:
        advantages.append("Good alternative rotation option")
    if not limitations:
        limitations.append("No major agro-climatic constraints observed")

    return {
        "crop": crop_name.capitalize(),
        "compatibility_score": max(25, min(95, score)),
        "advantages": advantages[:3],
        "limitations": limitations[:3],
        "requirements": f"pH: {ranges['ph'][0]}-{ranges['ph'][1]}, Temp: {ranges['temp'][0]}-{ranges['temp'][1]}°C, Rain: {ranges['rain'][0]}-{ranges['rain'][1]}mm",
        "regen_tip": ranges.get("regen_tip", "Practice balanced fertilization.")
    }


def train_and_export():
    dataset_rows = []
    regen_tips_dict = {}

    for crop, data in CROP_AGRONOMIC_RANGES.items():
        dataset_rows.extend(sample_crop_data(crop, data, n_samples=300))
        regen_tips_dict[crop] = data["regen_tip"]

    X = np.array([[row[f] for f in FEATURES] for row in dataset_rows], dtype=np.float32)
    y = np.array([row["label"] for row in dataset_rows])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    clf = RandomForestClassifier(
        n_estimators=180,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    print(f"Model validation accuracy: {score * 100:.2f}% across {len(CROP_AGRONOMIC_RANGES)} crops.")

    bundle = {
        "model": clf,
        "features": FEATURES,
        "crop_ranges": CROP_AGRONOMIC_RANGES,
        "regen_tips": regen_tips_dict,
        "classes": list(clf.classes_)
    }

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "crop_model.pkl")
    joblib.dump(bundle, out_path)

    # Also save in root backend directory for backward compatibility
    root_out_path = os.path.join(os.path.dirname(__file__), "crop_model.pkl")
    joblib.dump(bundle, root_out_path)

    print(f"Saved model bundle to {out_path} and {root_out_path}")


if __name__ == "__main__":
    train_and_export()
