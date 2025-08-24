# test.py

import pytest
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "model.joblib"
DATA_PATH = "heart_disease.csv"

@pytest.fixture
def model():
    return joblib.load(MODEL_PATH)

@pytest.fixture
def sample_data():
    df = pd.read_csv(DATA_PATH)
    # Simple data prep (as in train.py)
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    df['target'] = df['target'].map({'yes': 1, 'no': 0})
    df = df.fillna(df.median(numeric_only=True))
    if 'sno' in df.columns:
        df = df.drop(columns=['sno'])
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def test_model_can_load(model):
    assert model is not None, "Model did not load successfully"

def test_model_shape(model, sample_data):
    X, _ = sample_data
    pred = model.predict(X)
    assert len(pred) == len(X), "Prediction shape mismatch"

def test_model_accuracy(model, sample_data):
    X, y = sample_data
    score = model.score(X, y)
    assert score > 0.6, f"Test accuracy is too low: {score}"

def test_model_predict_single(model, sample_data):
    X, _ = sample_data
    result = model.predict([X.iloc[0]])
    assert result[0] in [0, 1], "Prediction is not 0 or 1"

def test_model_handles_nan(model, sample_data):
    X, _ = sample_data
    nan_row = X.iloc[0].copy()
    nan_row['chol'] = np.nan  # Insert missing value
    nan_row = nan_row.fillna(X['chol'].median())
    result = model.predict([nan_row])
    assert result[0] in [0, 1], "Model did not handle NaN correctly"
