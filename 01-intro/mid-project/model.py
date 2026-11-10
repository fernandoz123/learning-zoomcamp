from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier 

CATEGORICAL_FEATURES = (
    "gender",
    "hypertension",
    "heart_disease",
    "smoking_history",
)
NUMERICAL_FEATURES = (
    "bmi",
    "age",
    "blood_glucose_level",
    "HbA1c_level",
)
TARGET = "diabetes"
FEATURES = list(CATEGORICAL_FEATURES) + list(NUMERICAL_FEATURES)




def prepare_features(df):
    """
    Convert a DataFrame into a list of dictionaries
    suitable for DictVectorizer.
    """

    df_features = df[FEATURES].copy()

    # Convert categorical features to strings
    for col in CATEGORICAL_FEATURES:
        df_features[col] = df_features[col].astype(str)

    return df_features.to_dict(orient='records')

def build_pipeline():
    return make_pipeline(
        DictVectorizer(),
        XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=1,
        ),
    )
    
def split_data(df):
    """
    Split the dataset into train, validation and test sets.

    Proportions:
        Train: 60%
        Validation: 20%
        Test: 20%
    """

    # First split: 80% full train, 20% test
    df_full_train, df_test = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET]
    )

    # Second split: 75% train, 25% validation
    df_train, df_val = train_test_split(
        df_full_train,
        test_size=0.25,
        random_state=42,
        stratify=df_full_train[TARGET]
    )

    return df_train, df_val, df_test
