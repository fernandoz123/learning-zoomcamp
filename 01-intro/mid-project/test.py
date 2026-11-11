from pathlib import Path
import joblib


MODEL_PATH = Path(__file__).parent / "models" / "pipeline.joblib"
PACIENT = {
    "gender": "Male",
    "age": 68,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 31.5,
    "HbA1c_level": 6.0,
    "blood_glucose_level": 201
}

with MODEL_PATH.open("rb") as model_file:
    pipeline = joblib.load(model_file)


first = float(pipeline.predict_proba([PACIENT])[0, 1])
second = float(pipeline.predict_proba([PACIENT])[0, 1])

assert 0 < first < 1
assert first == second
print({"diabetes_probability": round(first, 3), "diabetes": first >= 0.5})