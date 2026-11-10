import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).parent / "data" / "pipeline.joblib"
PACIENT = {
    "gender": "Female",
    "age": 50,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 28.5,
    "HbA1c_level": 6.2,
    "blood_glucose_level": 140
}

with MODEL_PATH.open("rb") as model_file:
    pipeline = joblib.load(model_file)


first = float(pipeline.predict_proba([PACIENT])[0, 1])
second = float(pipeline.predict_proba([PACIENT])[0, 1])

assert 0 < first < 1
assert first == second
print({"diabetes_probability": round(first, 3), "diabetes": first >= 0.5})