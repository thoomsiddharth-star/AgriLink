"""
Satellite / environmental data service.

Currently provides demo data with clean service abstraction.
All data is clearly labeled as "Demo data".

To connect real providers:
  - Google Earth Engine (free tier) for NDVI, soil moisture, LST
  - Copernicus Open Access Hub for Sentinel-2 data
  - NASA MODIS for vegetation indices
"""

from typing import Optional
import random
import math

# Demo satellite data for known regions — labeled as demo data
DEMO_SATELLITE_DATA = {
    "telangana": {
        "ndvi": 0.68,
        "soil_moisture": 0.41,
        "land_surface_temp": 32.5,
        "vegetation_health": "Moderate",
        "evi": 0.45,  # Enhanced Vegetation Index
    },
    "andhra_pradesh": {
        "ndvi": 0.55,
        "soil_moisture": 0.35,
        "land_surface_temp": 34.2,
        "vegetation_health": "Moderate",
        "evi": 0.38,
    },
    "punjab": {
        "ndvi": 0.78,
        "soil_moisture": 0.48,
        "land_surface_temp": 28.8,
        "vegetation_health": "Good",
        "evi": 0.55,
    },
    "maharashtra": {
        "ndvi": 0.52,
        "soil_moisture": 0.33,
        "land_surface_temp": 35.1,
        "vegetation_health": "Moderate-Low",
        "evi": 0.35,
    },
    "default": {
        "ndvi": 0.60,
        "soil_moisture": 0.38,
        "land_surface_temp": 31.0,
        "vegetation_health": "Moderate",
        "evi": 0.40,
    },
}

# NDVI interpretation thresholds (well-established in remote sensing)
NDVI_THRESHOLDS = {
    "barren": (-1.0, 0.1),
    "sparse_vegetation": (0.1, 0.2),
    "low_vegetation": (0.2, 0.4),
    "moderate_vegetation": (0.4, 0.6),
    "dense_vegetation": (0.6, 0.8),
    "very_dense_vegetation": (0.8, 1.0),
}


def interpret_ndvi(ndvi: float) -> dict:
    """Interpret NDVI value into human-readable vegetation status."""
    if ndvi < 0.1:
        return {"status": "warning", "label": "Barren / No vegetation", "color": "red"}
    elif ndvi < 0.2:
        return {"status": "warning", "label": "Sparse vegetation", "color": "orange"}
    elif ndvi < 0.4:
        return {"status": "caution", "label": "Low vegetation", "color": "yellow"}
    elif ndvi < 0.6:
        return {"status": "good", "label": "Moderate vegetation", "color": "lightgreen"}
    elif ndvi < 0.8:
        return {"status": "good", "label": "Dense vegetation", "color": "green"}
    else:
        return {"status": "good", "label": "Very dense vegetation", "color": "darkgreen"}


def interpret_soil_moisture(sm: float) -> dict:
    """Interpret soil moisture fraction."""
    if sm < 0.15:
        return {"status": "warning", "label": "Very Dry", "color": "red"}
    elif sm < 0.25:
        return {"status": "caution", "label": "Dry", "color": "orange"}
    elif sm < 0.40:
        return {"status": "good", "label": "Adequate", "color": "green"}
    elif sm < 0.55:
        return {"status": "good", "label": "Moist", "color": "blue"}
    else:
        return {"status": "caution", "label": "Saturated", "color": "darkblue"}


def get_satellite_data(lat: float, lon: float, region: Optional[str] = None) -> dict:
    """
    Get satellite/environmental data for a location.

    Currently returns demo data — clearly labeled.
    The service abstraction is designed so that a real provider
    (Google Earth Engine, Copernicus, etc.) can be connected by
    implementing fetch_real_satellite_data() and toggling the source.
    """
    region_key = (region or "default").lower().replace(" ", "_")
    base = DEMO_SATELLITE_DATA.get(region_key, DEMO_SATELLITE_DATA["default"])

    # Add small variation based on coordinates to make demo more realistic
    lat_factor = math.sin(lat * 0.1) * 0.05
    lon_factor = math.cos(lon * 0.1) * 0.03

    ndvi = round(max(0, min(1, base["ndvi"] + lat_factor)), 2)
    soil_moisture = round(max(0, min(1, base["soil_moisture"] + lon_factor)), 2)
    lst = round(base["land_surface_temp"] + lat_factor * 10, 1)

    # Determine vegetation health from NDVI
    ndvi_status = interpret_ndvi(ndvi)
    moisture_status = interpret_soil_moisture(soil_moisture)

    # Overall health assessment
    if ndvi >= 0.6 and soil_moisture >= 0.30:
        overall_health = "Good"
        overall_status = "good"
    elif ndvi >= 0.4 and soil_moisture >= 0.20:
        overall_health = "Moderate"
        overall_status = "caution"
    else:
        overall_health = "Needs Attention"
        overall_status = "warning"

    return {
        "ndvi": ndvi,
        "soil_moisture": soil_moisture,
        "land_surface_temp": lst,
        "vegetation_health": overall_health,
        "evi": round(base.get("evi", ndvi * 0.7), 2),
        "data_source": "Demo data — connect Google Earth Engine or Copernicus for live satellite data",
        "data_quality": "demo",
        "interpretation": {
            "ndvi": ndvi_status,
            "soil_moisture": moisture_status,
            "overall": {"status": overall_status, "label": overall_health},
        },
        "coordinates": {"lat": lat, "lon": lon},
    }
