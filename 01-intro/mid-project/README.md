# ML Zoomcamp midterm project
# 1. Project Description
This project develops an end-to-end Machine Learning classification to predict the likelihood of diabetes in patients based on clinical indicators—including age, BMI, hypertension, heart disease, smoking history, blood glucose levels, and glycated hemoglobin (HbA1c)

# 2. Dataset
Dataset is taken from [kaggle diabetes dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).

The original data consists from 100 000 rows and 9 columns. I dropped 3,854 duplicated rows, that leaves us with 96,146 rows.
# 3. Project structure
```text
├── data/
│   ├── diabetes_prediction_dataset.csv
│   └── diabetes_prediction_dataset_cleaned_v2.csv
│   └── ...
├── models/
│   ├── model_metadata.json
│   ├── pipeline.joblib
│   └── ...
├── notebooks/
│   ├── notebook.ipynb
│   ├── train.ipynb
│   └── ...
├── .python_version
├── Dockerfile
├── model.py
├── predict.py
├── pyproject.toml
├── README.md
├── test.py
├── train.py
```

# 4. Deployment
## Run the project
This project uses uv for dependency management.

### Download the repository
[Folder repo mid-project]()
### To run the project  locally from your machine
Create the virtual enviroment, Install dependendencies and test model
```bash
uv sync --locked
uv run python test.py
```
### Start uvicorn server
```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
curl -s http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "gender": "Male",
    "age": 68,
    "hypertension": 1,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 31.5,
    "HbA1c_level": 6.0,
    "blood_glucose_level": 201
  }'
```
### Run the docker deployment 
```bash
docker build -t api-diabetes .
docker run -d -p 9696:9696 --name diabetes-container api-diabetes:1.0
```


