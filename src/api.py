from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Initialize the application
app = FastAPI(
    title="California Housing Prediction API",
    description="An MLOps API serving a Random Forest model",
    version="1.0"
)

# 1. Define the Input Schema
# These exactly match the columns from the California Housing dataset
class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Global variables to hold artifacts in memory
MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.joblib"
model = None
scaler = None

# 2. Load Artifacts on Startup
@app.on_event("startup")
def load_artifacts():
    global model, scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Artifacts missing in models/. Did you run src/train.py?")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully into memory.")

# 3. Define the Prediction Endpoint
@app.post("/predict")
def predict(features: HousingFeatures):
    try:
        # Convert the JSON payload into a DataFrame
        # We use a DataFrame so the column names match what the scaler expects
        input_data = pd.DataFrame([features.dict()])
        
        # Apply the exact same scaling transformation used in training
        scaled_features = scaler.transform(input_data)
        
        # Generate prediction
        prediction = model.predict(scaled_features)
        
        # The target variable is expressed in hundreds of thousands of dollars ($100,000s)
        return {
            "predicted_value_100k": round(float(prediction[0]), 3),
            "estimated_dollars": round(float(prediction[0]) * 100000, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
