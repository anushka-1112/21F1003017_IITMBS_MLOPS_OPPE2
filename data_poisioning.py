import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load model and data
model = joblib.load('model.joblib')
df = pd.read_csv("~/data.csv")  # Replace with your actual CSV filename

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

poison_frac = 0.2
indices_to_poison = np.random.choice(len(y_train), int(poison_frac * len(y_train)), replace=False)
y_train_poisoned = y_train.copy()
y_train_poisoned.iloc[indices_to_poison] = 1 - y_train_poisoned.iloc[indices_to_poison]  # Flip 0->1, 1->0
model_poisoned = RandomForestClassifier(random_state=42)
model_poisoned.fit(X_train, y_train_poisoned)

from sklearn.metrics import accuracy_score

y_pred_clean = model.predict(X_test)
y_pred_poisoned = model_poisoned.predict(X_test)

print(f"Clean model accuracy: {accuracy_score(y_test, y_pred_clean)}")
print(f"Poisoned model accuracy: {accuracy_score(y_test, y_pred_poisoned)}")
