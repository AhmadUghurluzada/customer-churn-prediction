import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Create the FastAPI app
app = FastAPI(title="Customer Churn Prediction API")

# Load model and scaler once when the server starts
model = joblib.load('churn_model.joblib')
scaler = joblib.load('churn_scaler.joblib')
training_columns = joblib.load('churn_columns.joblib')

class CustomerFeatures(BaseModel):
    SeniorCitizen: int
    tenure: float
    MonthlyCharges: float
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


# Health check endpoint
@app.get('/')
def home():
    return {'message': 'Customer Churn Prediction API is running'}

@app.post('/predict')
def predict(features: CustomerFeatures):
    
    # Build dataframe from input
    data = pd.DataFrame([features.dict()])
    
    # One-hot encode — same columns as training
    data = pd.get_dummies(data, 
                          columns=["gender", "Partner", "Dependents",
                                   "PhoneService", "MultipleLines",
                                   "InternetService", "OnlineSecurity",
                                   "OnlineBackup", "DeviceProtection",
                                   "TechSupport", "StreamingTV", 
                                   "StreamingMovies", "Contract",
                                   "PaperlessBilling", "PaymentMethod"],
                          drop_first=True,
                          dtype=int)
    
    # Add num_services feature
    service_cols = ['PhoneService_Yes', 'OnlineSecurity_Yes', 
                    'OnlineBackup_Yes', 'DeviceProtection_Yes',
                    'TechSupport_Yes', 'StreamingTV_Yes', 
                    'StreamingMovies_Yes']
    data['num_services'] = data[[c for c in service_cols if c in data.columns]].sum(axis=1)
    
    # Align columns with training data
    data = data.reindex(columns=training_columns, fill_value=0)
    
    # Scale and predict
    input_scaled = scaler.transform(data)
    proba = model.predict_proba(input_scaled)[:, 1][0]
    prediction = int(proba >= 0.3)
    
    return {
        'churn_probability': round(float(proba), 4),
        'will_churn': bool(prediction),
        'threshold_used': 0.3
    }