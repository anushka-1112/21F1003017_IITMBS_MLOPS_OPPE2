from fastapi import FastAPI, Request
import joblib
import pandas as pd

app = FastAPI()

# Load trained model
model = joblib.load('model.joblib')

@app.post("/predict")
async def predict(request: Request):
    json_data = await request.json()
    input_df = pd.DataFrame([json_data])
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}
