
import json
import hashlib
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from model import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    prepare_features,
    build_pipeline,
    split_data
)

#Rutas del proyecto
DATA_PATH = Path("data/diabetes_prediction_dataset_cleaned.csv")
MODEL_PATH = Path("models/pipeline.joblib")
METADATA_PATH = Path("models/model_metadata.json")

def get_file_hash(file_path):
    """Calcula el hash SHA256 del dataset."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

def evaluate_model(pipeline, df):
    X = prepare_features(df)
    y = df[TARGET]

    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, y_prob))
    }

    return metrics

def train_model(df):
    X_train = prepare_features(df)
    y_train = df[TARGET]

    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)

    return pipeline

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {DATA_PATH}"
        )
    # 2. Cargar dataset
    print("Cargando dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Registros: {len(df)}")

    # 3. Dividir dataset
    print("\nDividiendo dataset...")

    df_train, df_val, df_test = split_data(df)

    print(f"Entrenamiento: {len(df_train)}")
    print(f"Validación: {len(df_val)}")
    print(f"Prueba: {len(df_test)}")
    
    # 4. Entrenar con el conjunto de entrenamiento
    print("\nEntrenando modelo...")

    pipeline = train_model(df_train)

    # 5. Evaluar con validación
    print("\nEvaluación en test:")

    test_metrics = evaluate_model(pipeline, df_test)

    print(json.dumps(test_metrics, indent=4))

    # 8. Guardar modelo
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    
    # 9. Guardar metadatos
    metadata = {
        "model": "XGBoostClassifier",
        "target": TARGET,
        "features": {
            "numerical": list(NUMERICAL_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
        },
        "dataset": {
            "filename": DATA_PATH.name,
            "sha256": get_file_hash(DATA_PATH),
            "rows": len(df),
        },
        "split": {
            "train": len(df_train),
            "validation": len(df_val),
            "test": len(df_test),
        },
        "test_metrics": test_metrics,
        "random_state": 42,
    }
    
    with open(METADATA_PATH, "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nEntrenamiento finalizado.")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Metadatos guardados en: {METADATA_PATH}")

if __name__ == "__main__":
    main()