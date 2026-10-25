
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
saved = joblib.load(BASE_DIR / "churn_model.joblib")
model = saved["model"]
threshold = saved["threshold"]

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability using a trained ML pipeline.",
    version="1.0.0",
)


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    PaperlessBilling: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict_churn(customer: CustomerFeatures):
    customer_df = pd.DataFrame([customer.model_dump()])

    probability = float(model.predict_proba(customer_df)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "churn_probability": round(probability, 4),
        "prediction": prediction,
        "result": "Likely to churn" if prediction else "Likely to stay",
        "threshold": threshold,
    }
