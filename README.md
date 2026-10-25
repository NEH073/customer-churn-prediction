# Customer Churn Prediction using Machine Learning

## 🚀 Live Demo

Try the application here: [Customer Churn Prediction App](https://customer-churn-prediction-csqj96dngmlavnwust3gsg.streamlit.app/)

Enter customer details to estimate churn probability and explore how customer characteristics affect predictions.

## Project Overview

A machine learning project that predicts whether a telecom customer is likely to leave the service. It uses customer demographics, account information, subscription details, and service usage attributes to estimate churn probability.

The project includes an interactive Streamlit application and a REST API built with FastAPI, packaged using Docker.

## Business Problem

Customer churn affects recurring revenue and customer retention. This project aims to identify customers at risk of leaving so businesses can consider targeted retention strategies.

## Dataset

- IBM Telco Customer Churn dataset
- Includes customer demographics, services, contract details, billing information, and churn labels.
- Rows with missing `TotalCharges` values are removed during data cleaning.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Logistic Regression
- **API:** FastAPI, Pydantic, Uvicorn
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib
- **Deployment:** Docker
- **Interactive UI:** Streamlit

## Project Workflow

1. Data cleaning and exploratory data analysis.
2. Feature and target separation.
3. Stratified training, validation, and test split.
4. Numerical feature scaling and categorical feature encoding using a preprocessing pipeline.
5. Logistic Regression training with class weighting.
6. Classification threshold selection based on validation performance.
7. Model evaluation using precision, recall, F1-score, accuracy, and a confusion matrix.
8. Saving the trained pipeline for new customer predictions.
9. Exposing predictions through a FastAPI REST API.
10. Packaging the API and trained model in a Docker container.

## Model Evaluation

The Logistic Regression model uses a classification threshold of `0.30` to prioritize detecting customers who may churn.

| Metric | Test Result |
|---|---:|
| Precision | 41.9% |
| Recall | 92.2% |
| F1-score | 57.6% |
| Accuracy | 64.0% |

The model detected 345 of 374 customers who churned in the test set. Its high recall comes with false positives, meaning some customers predicted to churn actually stayed.

These results represent an initial baseline and do not guarantee real-world performance.

## API Endpoints

The FastAPI application exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Returns an API welcome message |
| GET | `/health` | Reports API health and model loading status |
| POST | `/predict` | Returns churn probability and the threshold-based prediction |

### Example Prediction Request

Send a `POST` request to `/predict` with customer features:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "PaperlessBilling": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.0,
  "TotalCharges": 140.0
}
```

Example response:

```json
{
  "churn_probability": 0.8471,
  "prediction": 1,
  "result": "Likely to churn",
  "threshold": 0.3
}
```

The probability above is an example from a tested request. `prediction: 1` indicates likely churn under the configured threshold.

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/NEH073/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Add the dataset

Place the IBM Telco Customer Churn dataset in the project root and name it `churn.csv`.

### 4. Train the model

```bash
python train.py
```

### 5. Launch the Streamlit application

```bash
python -m streamlit run app.py
```

### 6. Run the FastAPI application

Ensure the model file `churn_model.joblib` exists, then run:

```bash
python -m uvicorn api:app --reload
```

Open the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Run with Docker

Ensure Docker Desktop is running and the trained model file `churn_model.joblib` is present.

### 1. Build the image

```bash
docker build -t customer-churn-api .
```

### 2. Start the container

Port `8001` is used here to avoid conflicting with other applications that may already use port `8000`.

```bash
docker run --name customer-churn-container -p 8001:8000 customer-churn-api
```

### 3. Test the API

- API health: [http://127.0.0.1:8001/health](http://127.0.0.1:8001/health)
- Interactive API documentation: [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

Use `POST /predict` in the interactive documentation to submit customer details and receive a prediction.

To stop the running container, press `Ctrl + C`. To remove the stopped container later, run:

```bash
docker rm customer-churn-container
```

## Model Evaluation Visualizations

### Confusion Matrix

The confusion matrix shows how accurately the model identifies customers who stay and customers who churn. At the selected threshold, the model prioritizes detecting churners while accepting some false-positive predictions.

![Customer Churn Confusion Matrix](confusion_matrix.png)

### Feature Coefficients

This chart displays the strongest positive and negative Logistic Regression coefficients. Positive coefficients push predictions toward churn, while negative coefficients push predictions toward staying. These associations do not establish causation.

![Customer Churn Feature Coefficients](feature_importance.png)

## Future Improvements

- Compare additional classification algorithms.
- Improve the precision-recall trade-off.
- Add automated tests for API validation and prediction behavior.
- Add model monitoring and prediction logging.
- Explore model versioning and automated deployment workflows.
