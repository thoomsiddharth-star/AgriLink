"""
Soil data service.

Currently provides regional default estimates and accepts user-provided values.
All data is clearly labeled with its source (estimated / user-provided).

To connect real soil data:
  - Integrate with SoilGrids API (ISRIC): https://rest.isric.org
  - Or India's Soil Health Card portal data
  - Or FAO soil database
"""

from typing import Optional

# Regional soil defaults — approximate values from publicly available
# Indian agricultural data for major regions. These are ESTIMATES only.
REGIONAL_DEFAULTS = {
    "telangana": {
        "ph": 7.2,
        "nitrogen": 68,
        "phosphorus": 42,
        "potassium": 55,
        "moisture": 0.38,
        "soil_type": "Black Cotton Soil (Vertisol)",
        "organic_carbon": 0.52,
        "source": "Regional estimate based on Telangana agricultural survey data",
    },
    "andhra_pradesh": {
        "ph": 7.5,
        "nitrogen": 58,
        "phosphorus": 38,
        "potassium": 48,
        "moisture": 0.35,
        "soil_type": "Red Sandy Loam (Alfisol)",
        "organic_carbon": 0.45,
        "source": "Regional estimate based on AP agricultural survey data",
    },
    "punjab": {
        "ph": 8.0,
        "nitrogen": 95,
        "phosphorus": 55,
        "potassium": 62,
        "moisture": 0.42,
        "soil_type": "Alluvial Soil (Inceptisol)",
        "organic_carbon": 0.58,
        "source": "Regional estimate based on Punjab agricultural survey data",
    },
    "maharashtra": {
        "ph": 7.8,
        "nitrogen": 72,
        "phosphorus": 48,
        "potassium": 52,
        "moisture": 0.36,
        "soil_type": "Black Cotton Soil (Vertisol)",
        "organic_carbon": 0.50,
        "source": "Regional estimate based on Maharashtra agricultural survey data",
    },
    "karnataka": {
        "ph": 6.8,
        "nitrogen": 62,
        "phosphorus": 40,
        "potassium": 50,
        "moisture": 0.34,
        "soil_type": "Red Laterite Soil (Oxisol)",
        "organic_carbon": 0.48,
        "source": "Regional estimate based on Karnataka agricultural survey data",
    },
    "tamil_nadu": {
        "ph": 7.0,
        "nitrogen": 65,
        "phosphorus": 44,
        "potassium": 53,
        "moisture": 0.37,
        "soil_type": "Red Loam / Alluvial",
        "organic_carbon": 0.46,
        "source": "Regional estimate based on Tamil Nadu agricultural survey data",
    },
    "west_bengal": {
        "ph": 6.5,
        "nitrogen": 78,
        "phosphorus": 50,
        "potassium": 58,
        "moisture": 0.45,
        "soil_type": "Alluvial Soil (Entisol)",
        "organic_carbon": 0.55,
        "source": "Regional estimate based on West Bengal agricultural survey data",
    },
    "default": {
        "ph": 6.8,
        "nitrogen": 70,
        "phosphorus": 45,
        "potassium": 55,
        "moisture": 0.38,
        "soil_type": "Loam (general)",
        "organic_carbon": 0.50,
        "source": "General Indian agricultural estimate",
    },
}

# Soil health thresholds for status indicators
SOIL_THRESHOLDS = {
    "ph": {"low": 5.5, "optimal_low": 6.0, "optimal_high": 7.5, "high": 8.5},
    "nitrogen": {"low": 40, "medium": 80, "high": 120},
    "phosphorus": {"low": 20, "medium": 50, "high": 80},
    "potassium": {"low": 30, "medium": 60, "high": 90},
    "moisture": {"low": 0.20, "medium": 0.35, "high": 0.50},
    "organic_carbon": {"low": 0.40, "medium": 0.60, "high": 0.80},
}


def get_soil_status(param: str, value: float) -> dict:
    """Return a human-friendly status for a soil parameter."""
    thresholds = SOIL_THRESHOLDS.get(param)
    if not thresholds:
        return {"status": "unknown", "label": "Unknown", "color": "gray"}

    if param == "ph":
        if value < thresholds["low"]:
            return {"status": "warning", "label": "Too Acidic", "color": "orange"}
        elif value < thresholds["optimal_low"]:
            return {"status": "caution", "label": "Slightly Acidic", "color": "yellow"}
        elif value <= thresholds["optimal_high"]:
            return {"status": "good", "label": "Optimal", "color": "green"}
        elif value <= thresholds["high"]:
            return {"status": "caution", "label": "Slightly Alkaline", "color": "yellow"}
        else:
            return {"status": "warning", "label": "Too Alkaline", "color": "orange"}
    else:
        if value < thresholds["low"]:
            return {"status": "warning", "label": "Low", "color": "orange"}
        elif value < thresholds["medium"]:
            return {"status": "caution", "label": "Medium", "color": "yellow"}
        elif value <= thresholds["high"]:
            return {"status": "good", "label": "Good", "color": "green"}
        else:
            return {"status": "caution", "label": "High", "color": "yellow"}


def get_soil_data(region: Optional[str] = None, user_data: Optional[dict] = None) -> dict:
    """
    Get soil data for a region. Merges user-provided values over defaults.
    All values are clearly labeled with their source.
    """
    region_key = (region or "default").lower().replace(" ", "_")
    defaults = REGIONAL_DEFAULTS.get(region_key, REGIONAL_DEFAULTS["default"])

    result = {**defaults}
    result["data_quality"] = "estimated"

    if user_data:
        for key in ["ph", "nitrogen", "phosphorus", "potassium", "moisture",
                     "soil_type", "organic_carbon"]:
            if key in user_data and user_data[key] is not None:
                result[key] = user_data[key]
                result["data_quality"] = "user-provided"
                result["source"] = "User-provided measurements"

    # Add status indicators
    result["status"] = {
        "ph": get_soil_status("ph", result["ph"]),
        "nitrogen": get_soil_status("nitrogen", result["nitrogen"]),
        "phosphorus": get_soil_status("phosphorus", result["phosphorus"]),
        "potassium": get_soil_status("potassium", result["potassium"]),
        "moisture": get_soil_status("moisture", result["moisture"]),
    }

    return result
