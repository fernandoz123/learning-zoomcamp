import hashlib
import os
import joblib
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "models" / "pipeline.joblib"))
with MODEL_PATH.open("rb") as model_file:
    pipeline = joblib.load(model_file)

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")

class PatientData(BaseModel):   
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok",
            "model_hash": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()}

@app.post("/predict")
def predict(patient_data: PatientData) -> dict[str, Any]:
    """Predict diabetes probability for a given patient."""
    patient_dict = patient_data.dict()
    probability = float(pipeline.predict_proba([patient_dict])[0, 1])
    return {
        "diabetes_probability": round(probability, 3),
        "diabetes": probability >= 0.5
    }