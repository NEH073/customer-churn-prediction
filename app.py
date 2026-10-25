
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter a customer's details to estimate their likelihood of leaving the service."
)

@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")

try:
    saved = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Run train.py first to generate churn_model.joblib."
    )
    st.stop()

# Support either a saved dictionary or a directly saved pipeline.
if isinstance(saved, dict):
    model = saved.get("model", saved.get("pipeline"))
    threshold = saved.get("threshold", 0.30)
else:
    model = saved
    threshold = 0.30

if model is None:
    st.error("Could not find the trained model inside churn_model.joblib.")
    st.stop()

st.subheader("Customer details")

with st.form("customer_details"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Has partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple lines", ["No", "Yes", "No phone service"]
        )
        internet = st.selectbox(
            "Internet service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online backup", ["Yes", "No", "No internet service"]
        )

    with col2:
        device_protection = st.selectbox(
            "Device protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly charges ($)", min_value=0.0, value=70.0, step=1.0
        )
        total_charges = st.number_input(
            "Total charges ($)", min_value=0.0, value=840.0, step=10.0
        )

    submitted = st.form_submit_button("Predict churn", type="primary")

if submitted:
    customer = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple_lines,
                "InternetService": internet,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )

    try:
        probability = float(model.predict_proba(customer)[0][1])

        st.subheader("Prediction result")
        st.metric("Estimated churn probability", f"{probability:.1%}")
        st.progress(max(0.0, min(1.0, probability)))

        if probability >= threshold:
            st.warning("Likely to churn")
        else:
            st.success("Less likely to churn")

        st.caption(
            f"Decision threshold: {threshold:.0%}. "
            "This is a model estimate, not a guarantee."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.caption(
            "Check that the saved model expects these exact feature names "
            "and category values."
        )
