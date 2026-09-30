"""
Trains a lightweight crop-recommendation classifier.

The training data here is synthetic — generated from well-documented agronomic
ranges for each crop (N-P-K needs, temperature, humidity, soil pH, rainfall).
This is a reasonable placeholder for a hackathon demo. For a production /
judged submission, swap `CROP_RANGES` for a real dataset (e.g. an ICAR /
government agriculture dataset) and retrain the same way.

Run: python train_model.py
Produces: crop_model.pkl
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)

# Typical agronomic ranges per crop: N, P, K (kg/ha), temperature (C),
# humidity (%), soil pH, rainfall (mm). Ranges are approximate midpoints
# drawn from general agronomy references, not a scraped dataset.
CROP_RANGES = {
    "rice":       dict(N=(80, 120), P=(40, 60),  K=(40, 60),  temp=(22, 32), hum=(70, 90), ph=(5.5, 7.0), rain=(180, 300)),
    "wheat":      dict(N=(100, 140), P=(50, 70),  K=(40, 60),  temp=(12, 25), hum=(50, 70), ph=(6.0, 7.5), rain=(60, 110)),
    "maize":      dict(N=(80, 120), P=(40, 60),  K=(30, 50),  temp=(18, 30), hum=(55, 75), ph=(5.8, 7.0), rain=(80, 150)),
    "cotton":     dict(N=(90, 130), P=(30, 50),  K=(30, 50),  temp=(21, 35), hum=(50, 70), ph=(6.0, 8.0), rain=(60, 120)),
    "sugarcane":  dict(N=(150, 200), P=(60, 90),  K=(80, 120), temp=(24, 35), hum=(70, 90), ph=(6.0, 7.5), rain=(150, 250)),
    "chickpea":   dict(N=(15, 30),  P=(40, 60),  K=(15, 30),  temp=(15, 28), hum=(40, 60), ph=(6.0, 7.5), rain=(40, 80)),
    "groundnut":  dict(N=(15, 30),  P=(40, 60),  K=(40, 60),  temp=(22, 32), hum=(55, 75), ph=(5.5, 7.0), rain=(60, 120)),
    "mustard":    dict(N=(60, 90),  P=(30, 45),  K=(20, 35),  temp=(10, 22), hum=(40, 60), ph=(6.0, 7.5), rain=(30, 60)),
}

REGEN_TIPS = {
    "rice":      "Alternate wetting-and-drying irrigation; incorporate rice straw instead of burning it.",
    "wheat":     "Zero-till sowing into previous crop residue; legume rotation the following season.",
    "maize":     "Intercrop with legumes (cowpea/beans) for nitrogen fixation; retain crop residue as mulch.",
    "cotton":    "Cover crop in the off-season; drip irrigation to cut water use by up to 40%.",
    "sugarcane": "Trash mulching between rows; split nitrogen doses to reduce runoff.",
    "chickpea":  "Rhizobium seed inoculation; minimal tillage to preserve soil structure.",
    "groundnut": "Gypsum application at pegging stage; residue incorporation for organic matter.",
    "mustard":   "Line sowing with wider spacing to reduce disease pressure; avoid excess nitrogen (lodging risk).",
}

def sample_crop(crop, ranges, n=300):
    rows = []
    for _ in range(n):
        row = {
            "N": np.random.uniform(*ranges["N"]),
            "P": np.random.uniform(*ranges["P"]),
            "K": np.random.uniform(*ranges["K"]),
            "temperature": np.random.uniform(*ranges["temp"]),
            "humidity": np.random.uniform(*ranges["hum"]),
            "ph": np.random.uniform(*ranges["ph"]),
            "rainfall": np.random.uniform(*ranges["rain"]),
            "label": crop,
        }
        rows.append(row)
    return rows

def main():
    all_rows = []
    for crop, ranges in CROP_RANGES.items():
        all_rows.extend(sample_crop(crop, ranges))

    df = pd.DataFrame(all_rows)
    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X, y)

    joblib.dump({"model": clf, "regen_tips": REGEN_TIPS, "features": list(X.columns)}, "crop_model.pkl")
    print("Trained on", len(df), "synthetic samples across", len(CROP_RANGES), "crops.")
    print("Saved crop_model.pkl")

if __name__ == "__main__":
    main()
