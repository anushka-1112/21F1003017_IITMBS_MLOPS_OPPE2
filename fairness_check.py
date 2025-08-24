from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_negative_rate
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Load model and raw data
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
y_true = y_test  # True labels
y_pred = model.predict(X_test)
sensitive_feature = X_test['gender']  # or gender encoded
metrics = {
    'accuracy': accuracy_score,
    'selection_rate': selection_rate,
    'true_positive_rate': true_positive_rate,
    'false_negative_rate': false_negative_rate
}

metric_frame = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_feature)
print(metric_frame.by_group)

