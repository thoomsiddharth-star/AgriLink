"""
AgriLink SQLite Database Layer
Lightweight, zero-dependency persistence for farms, observations, advisories, and disease scans.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agrilink.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Farms table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS farms (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        farmer_name TEXT DEFAULT 'Farmer',
        region TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        area_acres REAL DEFAULT 5.0,
        soil_type TEXT DEFAULT 'Black Cotton Soil',
        created_at TEXT NOT NULL
    )
    """)

    # Observations table (Weather, Soil, Satellite)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farm_id TEXT NOT NULL,
        temperature REAL,
        humidity REAL,
        rainfall REAL,
        ph REAL,
        nitrogen REAL,
        phosphorus REAL,
        potassium REAL,
        soil_moisture REAL,
        ndvi REAL,
        timestamp TEXT NOT NULL,
        FOREIGN KEY (farm_id) REFERENCES farms (id)
    )
    """)

    # Advisories table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS advisories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farm_id TEXT NOT NULL,
        recommended_crop TEXT NOT NULL,
        confidence REAL NOT NULL,
        reasons TEXT,
        advisory_text TEXT,
        language TEXT DEFAULT 'en',
        created_at TEXT NOT NULL,
        FOREIGN KEY (farm_id) REFERENCES farms (id)
    )
    """)

    # Disease scans table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS disease_scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farm_id TEXT,
        image_name TEXT,
        crop_name TEXT,
        disease_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        severity TEXT,
        recommended_action TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # Seed default farm if empty
    cursor.execute("SELECT COUNT(*) as cnt FROM farms")
    if cursor.fetchone()["cnt"] == 0:
        cursor.execute("""
        INSERT INTO farms (id, name, farmer_name, region, latitude, longitude, area_acres, soil_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "farm-telangana-01",
            "Sri Venkateswara Organic Farm",
            "Ramesh Reddy",
            "Telangana",
            17.3850,
            78.4867,
            4.5,
            "Black Cotton Soil (Vertisol)",
            datetime.now().isoformat()
        ))
        cursor.execute("""
        INSERT INTO farms (id, name, farmer_name, region, latitude, longitude, area_acres, soil_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "farm-punjab-02",
            "Golden Harvest Farm",
            "Gurpreet Singh",
            "Punjab",
            30.9010,
            75.8573,
            8.0,
            "Alluvial Soil (Inceptisol)",
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()


def list_farms() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM farms ORDER BY created_at DESC").fetchall()
    farms = [dict(r) for r in rows]
    conn.close()
    return farms


def save_farm(farm_data: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_db_connection()
    farm_id = farm_data.get("id") or f"farm-{int(datetime.now().timestamp())}"
    now = datetime.now().isoformat()
    conn.execute("""
    INSERT OR REPLACE INTO farms (id, name, farmer_name, region, latitude, longitude, area_acres, soil_type, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        farm_id,
        farm_data.get("name", "New Farm"),
        farm_data.get("farmer_name", "Farmer"),
        farm_data.get("region", "Telangana"),
        float(farm_data.get("latitude", 17.3850)),
        float(farm_data.get("longitude", 78.4867)),
        float(farm_data.get("area_acres", 5.0)),
        farm_data.get("soil_type", "Loam"),
        now
    ))
    conn.commit()
    conn.close()
    return {**farm_data, "id": farm_id, "created_at": now}


def log_advisory(farm_id: str, recommended_crop: str, confidence: float, reasons: List[str], advisory_text: str, language: str = 'en'):
    conn = get_db_connection()
    conn.execute("""
    INSERT INTO advisories (farm_id, recommended_crop, confidence, reasons, advisory_text, language, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        farm_id,
        recommended_crop,
        confidence,
        json.dumps(reasons),
        advisory_text,
        language,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def log_disease_scan(farm_id: Optional[str], image_name: str, crop_name: str, disease_name: str, confidence: float, severity: str, action: str):
    conn = get_db_connection()
    conn.execute("""
    INSERT INTO disease_scans (farm_id, image_name, crop_name, disease_name, confidence, severity, recommended_action, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        farm_id or "default",
        image_name,
        crop_name,
        disease_name,
        confidence,
        severity,
        action,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def get_recent_history(limit: int = 10) -> Dict[str, Any]:
    conn = get_db_connection()
    advisories = [dict(r) for r in conn.execute("SELECT * FROM advisories ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()]
    for a in advisories:
        if a.get("reasons"):
            try:
                a["reasons"] = json.loads(a["reasons"])
            except Exception:
                pass
    scans = [dict(r) for r in conn.execute("SELECT * FROM disease_scans ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()]
    conn.close()
    return {"advisories": advisories, "disease_scans": scans}
