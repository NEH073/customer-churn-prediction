
import joblib
import pandas as pd

# Load the saved model and threshold
saved = joblib.load("churn_model.joblib")
model = saved["model"]
threshold = saved["threshold"]

# Example customer
customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 140.0,
}])

# Predict churn probability
probability = model.predict_proba(customer)[0, 1]
prediction = int(probability >= threshold)

print(f"Churn probability: {probability:.1%}")
print("Prediction:", "Likely to churn" if prediction else "Likely to stay")
