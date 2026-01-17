# MLOps Heart Disease Prediction Project

**Student ID:** 21F1003017  
**Course:** IIT Madras BS MLOps OPPE2  
**Date:** January 17, 2026

## Project Overview

This project implements a comprehensive Machine Learning Operations (MLOps) pipeline for heart disease prediction. It demonstrates end-to-end ML lifecycle management including model training, deployment, monitoring, fairness assessment, explainability, drift detection, and security considerations through data poisoning analysis.

## Features

- **Model Training**: Train a machine learning model for heart disease prediction using the provided dataset
- **API Deployment**: FastAPI-based REST API for real-time predictions
- **Batch Prediction**: Script for processing multiple predictions with logging and observability
- **Fairness Analysis**: Evaluate model fairness using Fairlearn with gender as sensitive attribute
- **Explainability**: Implement SHAP-based explainability for model predictions
- **Drift Detection**: Kolmogorov-Smirnov test for detecting data drift
- **Data Poisoning**: Demonstrate adversarial attacks through label interchange
- **Containerization**: Docker-based deployment for portability
- **Kubernetes Orchestration**: Auto-scaling deployment with HPA (max 3 pods)
- **Load Testing**: Performance monitoring with Locust
- **Comprehensive Testing**: Unit tests and sanity checks

## Project Structure

```
├── batch_predict.py          # Batch prediction script with logging
├── data_poisoning.py         # Data poisoning attack demonstration
├── Dockerfile                # Docker image configuration
├── drift_analysis_ks.py      # Drift detection using KS test
├── explainability.py         # SHAP-based model explainability
├── fairness_check.py         # Fairness evaluation with Fairlearn
├── heart_fastapi.py          # FastAPI application for predictions
├── locustfile.py             # Load testing configuration
├── readme.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── test.py                   # Model sanity tests
├── train.py                  # Model training script
├── data/
│   └── data.csv             # Original heart disease dataset
└── k8s/
    ├── deployment.yaml      # Kubernetes deployment manifest
    ├── hpa.yaml            # Horizontal Pod Autoscaler config
    └── service.yaml        # Kubernetes service for load balancing
```

## File Code Summaries

### Core Scripts

- **train.py**: Loads heart disease dataset, preprocesses features (encodes gender, fills missing values), trains a RandomForestClassifier, evaluates accuracy on test set, and saves the model as `model.joblib`.

- **heart_fastapi.py**: Implements a FastAPI web service for heart disease prediction with OpenTelemetry tracing, structured JSON logging, health checks (/live_check, /ready_check), and a /predict endpoint that preprocesses input data and returns predictions with trace IDs.

- **batch_predict.py**: Generates 100 random heart disease samples, sends them to the prediction API (via API_URL env var), collects responses with latency metrics, and saves results to `batch_prediction_results.json` for observability.

- **test.py**: Contains unit tests for model loading, prediction shape validation, accuracy threshold checks, single prediction functionality, and NaN handling using pytest.

### Analysis Scripts

- **fairness_check.py**: Loads the trained model and test data, uses Fairlearn's MetricFrame to compute fairness metrics (accuracy, selection rate, true positive rate, false negative rate) grouped by gender as the sensitive attribute.

- **explainability.py**: Loads the model and test data, uses SHAP TreeExplainer to compute SHAP values for predictions, and calculates mean absolute SHAP values to rank feature importance for model interpretability.

- **drift_analysis_ks.py**: Loads training data and generates 100 new random samples, preprocesses both datasets, and performs Kolmogorov-Smirnov tests on each feature to detect statistical drift between training and new data distributions.

- **data_poisoning.py**: Demonstrates data poisoning by loading the model and data, randomly flipping 20% of training labels, retraining a poisoned model, and comparing accuracies on the test set to show the impact of adversarial attacks.

### Configuration Files

- **Dockerfile**: Defines the Docker image build process for containerizing the FastAPI application.

- **requirements.txt**: Lists Python dependencies including FastAPI, scikit-learn, joblib, pandas, numpy, shap, fairlearn, and other required packages.

- **locustfile.py**: Configuration file for Locust load testing (currently empty, needs implementation for performance testing scenarios).

- **k8s/deployment.yaml**: Kubernetes deployment manifest for deploying the containerized application with resource limits and replica count.

- **k8s/hpa.yaml**: Horizontal Pod Autoscaler configuration for auto-scaling pods based on CPU utilization (max 3 pods).

- **k8s/service.yaml**: Kubernetes service definition for load balancing traffic to the deployed pods.

- **data/data.csv**: Original heart disease dataset used for training and evaluation.

## Installation & Setup

1. **Clone the repository** (if applicable) and navigate to the project directory.

2. **Install dependencies**:
   `bash
   pip install -r requirements.txt
   `

3. **Prepare data**: Ensure data/data.csv contains the heart disease dataset.

## Usage

### Training the Model
`bash
python train.py
`

### Running the API
`bash
python heart_fastapi.py
`

### Batch Prediction
`bash
python batch_predict.py
`

### Testing
`bash
python test.py
`

## Deployment

### Docker
Build and run the Docker container:
`bash
docker build -t heart-disease-mlops .
docker run -p 8000:8000 heart-disease-mlops
`

### Kubernetes
Deploy to Kubernetes cluster:
`bash
kubectl apply -f k8s/
`

The deployment includes:
- Horizontal Pod Autoscaler (max 3 pods)
- Load balancing service
- Auto-scaling based on CPU utilization

## Analysis & Evaluation

### Fairness Check
`bash
python fairness_check.py
`
Evaluates model fairness using gender as the sensitive attribute.

### Explainability
`bash
python explainability.py
`
Generates SHAP explanations for model predictions.

### Drift Analysis
`bash
python drift_analysis_ks.py
`
Detects data drift using Kolmogorov-Smirnov test.

### Data Poisoning
`bash
python data_poisoning.py
`
Demonstrates the impact of label interchange attacks on model performance.

## Load Testing

Use Locust for performance testing:
`bash
locust -f locustfile.py
`

## Technologies Used

- **Python**: Core programming language
- **FastAPI**: Web framework for API development
- **Scikit-learn**: Machine learning library
- **Fairlearn**: Fairness assessment
- **SHAP**: Model explainability
- **Docker**: Containerization
- **Kubernetes**: Container orchestration
- **Locust**: Load testing

## Key Deliverables

- Dockerized, API-deployed model on GCP
- Kubernetes deployment with auto-scaling
- Per-sample prediction with logging and observability
- Performance monitoring with high concurrency workloads
- Fairness, explainability, and drift analysis
- Data poisoning demonstration

## Notes

- Uses 100-row randomly generated test data for batch predictions and load testing
- Demonstrates production-ready MLOps practices
- Includes comprehensive monitoring and observability features
