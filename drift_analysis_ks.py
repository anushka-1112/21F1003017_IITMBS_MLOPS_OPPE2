# drift_analysis_ks.py

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder

def generate_random_samples(n=100):
    np.random.seed(42)
    samples = []
    for _ in range(n):
        sample = {
            'age': int(np.random.randint(30, 80)),
            'gender': np.random.choice(['male', 'female']),
            'cp': int(np.random.randint(0, 4)),
            'trestbps': float(np.random.uniform(90, 180)),
            'chol': float(np.random.uniform(150, 400)),
            'fbs': int(np.random.choice([0, 1])),
            'restecg': int(np.random.randint(0, 3)),
            'thalach': float(np.random.uniform(100, 210)),
            'exang': int(np.random.choice([0, 1])),
            'oldpeak': float(np.random.uniform(0.0, 6.0)),
            'slope': int(np.random.randint(0, 3)),
            'ca': int(np.random.randint(0, 5)),
            'thal': int(np.random.randint(0, 4))
        }
        samples.append(sample)
    return pd.DataFrame(samples)

# Load your training data
train_df = pd.read_csv('~/data.csv')

# Generate new 100 samples for prediction
new_data_df = generate_random_samples()

# Encode gender consistently
le_gender = LabelEncoder()
train_df['gender'] = le_gender.fit_transform(train_df['gender'])
new_data_df['gender'] = le_gender.transform(new_data_df['gender'])

# Impute missing values
train_df = train_df.fillna(train_df.median(numeric_only=True))
new_data_df = new_data_df.fillna(new_data_df.median(numeric_only=True))

# Features to analyze
features = ['age', 'gender', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

print("Feature-wise KS Test Results:")
print("-----------------------------")

for feature in features:
    train_vals = train_df[feature].values
    new_vals = new_data_df[feature].values
    
    ks_stat, p_value = ks_2samp(train_vals, new_vals)
    
    drift = "Yes" if p_value < 0.05 else "No"
    
    print(f"Feature: {feature}")
    print(f" KS Statistic: {ks_stat:.4f}")
    print(f" P-Value: {p_value:.4f}")
    print(f" Significant Drift Detected? {drift}")
    print("")

