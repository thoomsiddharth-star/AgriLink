"""
Interoperability Engine & AgriData Standard
Facilitates seamless data exchange between different regional and national agricultural systems.
Inspired by BRICS AgriN, OGC Land & Agriculture DWG, and UN FAO Digital Public Goods.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json


AGRIDATA_SCHEMA_SPEC = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AgriData Interoperable Standard",
    "version": "1.0.0",
    "description": "Standardized schema for exchanging weather, soil, satellite, and crop advisories across national and regional agricultural platforms.",
    "type": "object",
    "required": ["metadata", "location", "observations"],
    "properties": {
        "metadata": {
            "type": "object",
            "required": ["source_country", "source_organization", "standard_version", "timestamp"],
            "properties": {
                "source_country": {"type": "string", "example": "IND"},
                "source_region": {"type": "string", "example": "Telangana"},
                "source_organization": {"type": "string", "example": "ICAR-AgriLink-Node-01"},
                "standard_version": {"type": "string", "example": "1.0.0"},
                "timestamp": {"type": "string", "format": "date-time"}
            }
        },
        "location": {
            "type": "object",
            "required": ["latitude", "longitude"],
            "properties": {
                "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                "longitude": {"type": "number", "minimum": -180, "maximum": 180},
                "elevation_m": {"type": "number"},
                "farm_identifier": {"type": "string"}
            }
        },
        "observations": {
            "type": "object",
            "properties": {
                "soil": {
                    "type": "object",
                    "properties": {
                        "ph": {"type": "number", "minimum": 0, "maximum": 14},
                        "nitrogen_kg_ha": {"type": "number"},
                        "phosphorus_kg_ha": {"type": "number"},
                        "potassium_kg_ha": {"type": "number"},
                        "soil_moisture_fraction": {"type": "number", "minimum": 0, "maximum": 1},
                        "soil_type": {"type": "string"},
                        "data_quality": {"type": "string", "enum": ["measured", "estimated", "user-provided"]}
                    }
                },
                "weather": {
                    "type": "object",
                    "properties": {
                        "temperature_c": {"type": "number"},
                        "relative_humidity_pct": {"type": "number", "minimum": 0, "maximum": 100},
                        "precipitation_mm": {"type": "number", "minimum": 0},
                        "wind_speed_kmh": {"type": "number"},
                        "condition_code": {"type": "integer"}
                    }
                },
                "remote_sensing": {
                    "type": "object",
                    "properties": {
                        "ndvi": {"type": "number", "minimum": -1, "maximum": 1},
                        "evi": {"type": "number"},
                        "land_surface_temp_c": {"type": "number"},
                        "satellite_source": {"type": "string"}
                    }
                }
            }
        },
        "advisory": {
            "type": "object",
            "properties": {
                "recommended_crop": {"type": "string"},
                "suitability_score": {"type": "number", "minimum": 0, "maximum": 1},
                "reasons": {"type": "array", "items": {"type": "string"}},
                "regenerative_practice": {"type": "string"}
            }
        }
    }
}


def build_standard_record(
    country_code: str,
    region: str,
    lat: float,
    lon: float,
    soil_data: Dict[str, Any],
    weather_data: Dict[str, Any],
    satellite_data: Dict[str, Any],
    advisory_data: Optional[Dict[str, Any]] = None,
    org_name: str = "AgriLink-Node"
) -> Dict[str, Any]:
    """Transforms raw heterogeneous system data into the strict standard AgriData schema."""
    current_weather = weather_data.get("current", {})
    return {
        "metadata": {
            "source_country": country_code.upper(),
            "source_region": region,
            "source_organization": f"{org_name}-{country_code.upper()}",
            "standard_version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "interoperability_protocol": "AgriData/REST/v1"
        },
        "location": {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "farm_identifier": f"FARM-{region.upper()[:3]}-{abs(int(lat*100))}"
        },
        "observations": {
            "soil": {
                "ph": soil_data.get("ph"),
                "nitrogen_kg_ha": soil_data.get("nitrogen"),
                "phosphorus_kg_ha": soil_data.get("phosphorus"),
                "potassium_kg_ha": soil_data.get("potassium"),
                "soil_moisture_fraction": soil_data.get("moisture"),
                "soil_type": soil_data.get("soil_type", "Standard Loam"),
                "data_quality": soil_data.get("data_quality", "estimated")
            },
            "weather": {
                "temperature_c": current_weather.get("temperature"),
                "relative_humidity_pct": current_weather.get("humidity"),
                "precipitation_mm": current_weather.get("precipitation", 0.0),
                "wind_speed_kmh": current_weather.get("wind_speed"),
                "condition": current_weather.get("condition", "Clear sky")
            },
            "remote_sensing": {
                "ndvi": satellite_data.get("ndvi"),
                "soil_moisture_index": satellite_data.get("soil_moisture"),
                "land_surface_temp_c": satellite_data.get("land_surface_temp"),
                "vegetation_health": satellite_data.get("vegetation_health"),
                "satellite_source": satellite_data.get("data_source", "Demo/GEE")
            }
        },
        "advisory": {
            "recommended_crop": advisory_data.get("recommended_crop") if advisory_data else None,
            "confidence": advisory_data.get("confidence") if advisory_data else None,
            "reasons": advisory_data.get("reasons", []) if advisory_data else [],
            "regenerative_tip": advisory_data.get("regenerative_tip") if advisory_data else None
        } if advisory_data else None
    }


def validate_agridata_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validates an incoming external agricultural payload against AgriData standards."""
    errors = []
    if "metadata" not in payload:
        errors.append("Missing required root field: 'metadata'")
    else:
        for field in ["source_country", "timestamp"]:
            if field not in payload["metadata"]:
                errors.append(f"Missing required metadata field: '{field}'")

    if "location" not in payload:
        errors.append("Missing required root field: 'location'")
    else:
        if "latitude" not in payload["location"] or "longitude" not in payload["location"]:
            errors.append("Location must include 'latitude' and 'longitude'")

    if "observations" not in payload:
        errors.append("Missing required root field: 'observations'")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "schema_version": "1.0.0",
        "verified_at": datetime.utcnow().isoformat() + "Z"
    }


def get_sample_crossborder_records() -> Dict[str, Any]:
    """Sample records demonstrating multi-country interoperability (India, Brazil, Kenya)."""
    return {
        "records": [
            {
                "country": "India (Telangana)",
                "node": "ICAR-AgriLink-IND",
                "crop": "Rice / Cotton",
                "ndvi": 0.68,
                "soil_ph": 7.2,
                "status": "Transmitted & Verified"
            },
            {
                "country": "Brazil (Mato Grosso)",
                "node": "EMBRAPA-AgriLink-BRA",
                "crop": "Soybean / Maize",
                "ndvi": 0.74,
                "soil_ph": 6.1,
                "status": "Transmitted & Verified"
            },
            {
                "country": "Kenya (Rift Valley)",
                "node": "KALRO-AgriLink-KEN",
                "crop": "Maize / Sorghum",
                "ndvi": 0.58,
                "soil_ph": 6.5,
                "status": "Transmitted & Verified"
            }
        ]
    }
