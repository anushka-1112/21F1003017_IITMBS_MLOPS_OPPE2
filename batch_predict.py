# scripts/batch_predict.py

import os
import requests
import json
import time
import numpy as np

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
    return samples

def main():
    api_url = os.getenv('API_URL')
    if not api_url:
        print("ERROR: API_URL environment variable is not set")
        return

    samples = generate_random_samples(100)
    results = []
    
    print(f"Sending {len(samples)} samples to {api_url}")

    for i, sample in enumerate(samples):
        start_time = time.time()
        try:
            response = requests.post(api_url, json=sample, timeout=10)
            latency_ms = (time.time() - start_time) * 1000
            if response.ok:
                data = response.json()
                result = {
                    'sample_index': i+1,
                    'input': sample,
                    'predicted_class': data.get('predicted_class'),
                    'trace_id': data.get('trace_id'),
                    'latency_ms': latency_ms,
                    'status': 'success'
                }
                print(f"Sample {i+1}: Prediction={result['predicted_class']}, Latency={latency_ms:.2f}ms, TraceID={result['trace_id']}")
            else:
                result = {
                    'sample_index': i+1,
                    'input': sample,
                    'status': 'failed',
                    'http_status': response.status_code,
                    'latency_ms': latency_ms
                }
                print(f"Sample {i+1}: Failed with status {response.status_code}, Latency={latency_ms:.2f}ms")
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            result = {
                'sample_index': i+1,
                'input': sample,
                'status': 'error',
                'error': str(e),
                'latency_ms': latency_ms
            }
            print(f"Sample {i+1}: Exception {str(e)}, Latency={latency_ms:.2f}ms")
        results.append(result)

    # Save all results to file for reference
    with open('batch_prediction_results.jsonl', 'w') as f:
        for entry in results:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    main()
