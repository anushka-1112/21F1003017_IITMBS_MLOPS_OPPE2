import shap
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load your trained model
model = joblib.load('model.joblib')

# Load your test data (replace with your actual test data path)
X_test = pd.read_csv('~/data.csv')

# Preprocess test data: encode categorical variables as done during training
X_test['gender'] = X_test['gender'].map({'male': 1, 'female': 0})

# Handle missing values (if any)
X_test = X_test.fillna(X_test.median(numeric_only=True))

# Align columns with model feature names if applicable
model_features = model.feature_names_in_
X_test = X_test[model_features]

# Create SHAP explainer and compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Extract SHAP values for positive class (heart disease = 1)
shap_values_pos = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values[1]

# Calculate mean absolute SHAP value per feature for importance
feature_importance = pd.DataFrame({
    'feature': X_test.columns,
    'mean_abs_shap': np.mean(np.abs(shap_values_pos), axis=0)
}).sort_values(by='mean_abs_shap', ascending=False)

print(feature_importance)

