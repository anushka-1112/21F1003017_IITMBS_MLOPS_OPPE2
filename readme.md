cd.yaml - dockerized, API-deployed model execution on GCP. Use k8s with auto scaling (max pod-3) as a deployment layer, Demonstrating per sample prediction along with logging, observability. Use a 100-row randomly generated data for this task, Performance monitoring and request timeout analysis with a high concurrency workload using wrk. You can use the same random sample as generated in the earlier step. 

data/data.csv - original data
deployment.yaml - deployment of docker image
service.yaml - service for load balancing
hpa.yaml - for horizontal pods scaling
batch_predict.py - script for Demonstrating per sample prediction along with logging, observability. Use a 100-row randomly generated data for this task
data_poisioning.py - data poisoning attack using a simple label interchange and compare performance.
Dockerfile - to create docker image
drift_analysis.py - for drift analysis
explainability.py - explainability using mean shap values
fairness_check.py - fairness with “gender” as the sensitive attribute using fairlearn.  
heart_fastapi.py - fastapi app for hosting
requirements.txt - contains library versions
test.py - sanity test of model
train.py - training model