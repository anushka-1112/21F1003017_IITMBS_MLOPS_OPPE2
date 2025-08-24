from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging
import time
import json
import pandas as pd
import joblib
import sys

# --- OpenTelemetry imports ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# --- Setup Tracer ---
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# --- Setup Structured Logging ---
logger = logging.getLogger("heart-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- FastAPI App ---
app = FastAPI(title="Heart Disease Classifier API with Tracing & Logging")

# --- App state ---
app_state = {"is_ready": False, "is_alive": True}
model = None

# --- Model Load ---
@app.on_event("startup")
async def startup_event():
    global model
    try:
        time.sleep(2)  # simulate load latency
        model = joblib.load("model.joblib")
        app_state["is_ready"] = True
        logger.info(json.dumps({"event": "model_loaded", "status": "ready"}))
    except Exception as e:
        logger.exception(json.dumps({"event": "model_load_error", "error": str(e)}))
        app_state["is_ready"] = False

# --- Input Schema (Heart dataset features) ---
class HeartInput(BaseModel):
    age: int = Field(..., title="Age")
    gender: str = Field(..., title="Gender", description="male or female")
    cp: int = Field(..., title="Chest pain type")
    trestbps: float | None = Field(None, title="Resting blood pressure")
    chol: float | None = Field(None, title="Serum cholesterol")
    fbs: int = Field(..., title="Fasting blood sugar")
    restecg: int = Field(..., title="Resting ECG results")
    thalach: float | None = Field(None, title="Maximum heart rate achieved")
    exang: int = Field(..., title="Exercise induced angina")
    oldpeak: float = Field(..., title="ST depression induced by exercise")
    slope: int = Field(..., title="Slope of the peak exercise ST segment")
    ca: int = Field(..., title="Number of major vessels")
    thal: int = Field(..., title="Thalassemia type")

# --- Health Probes ---
@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=500)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=503)

# --- Middleware: Track request time ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

# --- Exception Handler ---
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "Welcome to Heart Disease Classifier API with Tracing & Logging"}

# --- Helper Function: feature preprocessing ---
def preprocess_features(data: HeartInput) -> pd.DataFrame:
    # Convert input to DataFrame and preprocess for prediction
    input_dict = data.dict()
    # Gender encoding
    input_dict['gender'] = 1 if input_dict['gender'].lower() == 'male' else 0
    # Fill missing values with model input defaults or zeros
    for k, v in input_dict.items():
        if v is None:
            input_dict[k] = 0
    return pd.DataFrame([input_dict])

# --- Prediction Endpoint ---
@app.post("/predict/", tags=["Prediction"])
async def predict_heart_disease(data: HeartInput, request: Request):
    with tracer.start_as_current_span("model_prediction") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        if not app_state["is_ready"] or model is None:
            logger.error(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": "Model not loaded or not ready"
            }))
            raise HTTPException(status_code=503, detail="Model not ready")

        try:
            input_df = preprocess_features(data)
            prediction = model.predict(input_df)[0]
            latency = round((time.time() - start_time) * 1000, 2)

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": data.dict(),
                "predicted_class": int(prediction),
                "latency_ms": latency,
                "status": "success"
            }))
            return {"predicted_class": int(prediction), "trace_id": trace_id}

        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "input": data.dict(),
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
