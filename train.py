# train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# --- Step 1: Load Data ---
df = pd.read_csv("heart_disease.csv")  # Replace with your actual CSV filename

# --- Step 2: Preprocess Data ---
# Map 'gender' and 'target' to numeric
df['gender'] = df['gender'].map({'male': 1, 'female': 0})
df['target'] = df['target'].map({'yes': 1, 'no': 0})

# Fill missing numeric values (simple strategy, replace with domain-specific as needed)
df = df.fillna(df.median(numeric_only=True))

# Drop non-feature columns if present
if 'sno' in df.columns:
    df = df.drop(columns=['sno'])

# --- Step 3: Train/Test Split ---
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 5: Evaluation ---
score = model.score(X_test, y_test)
print(f"Test accuracy: {score:.4f}")

# --- Step 6: Save Model ---
joblib.dump(model, "model.joblib")
print("Model saved as model.joblib")
