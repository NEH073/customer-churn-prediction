# Customer Churn Prediction using Machine Learning

## Project Overview
A machine learning project that predicts whether a telecom customer is likely to leave the service. The project uses customer account information, subscription details, and service usage attributes to estimate churn probability.

## Business Problem
Customer churn affects recurring revenue and customer retention. This project aims to identify customers at risk of leaving so that businesses can consider targeted retention strategies.

## Dataset
- IBM Telco Customer Churn dataset
- Contains customer demographics, services, contract details, billing information, and churn labels.
- Rows with missing `TotalCharges` values are removed during data cleaning.

## Tech Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib

## Project Workflow
1. Data cleaning and exploratory data analysis
2. Feature and target separation
3. Stratified training, validation, and test split
4. Numerical feature scaling and categorical feature encoding
5. Logistic Regression training with class weighting
6. Classification threshold selection based on validation performance
7. Model evaluation using precision, recall, F1-score, and a confusion matrix
8. Saving the trained pipeline for new customer predictions

## Model Evaluation
The current Logistic Regression model uses a classification threshold of 0.30 to prioritize detecting customers who may churn.

| Metric | Test Result |
|---|---:|
| Precision | 41.9% |
| Recall | 92.2% |
| F1-score | 57.6% |
| Accuracy | 64.0% |

The model detected 345 of 374 customers who churned in the test set. Its high recall comes with false positives, meaning some customers predicted to churn actually stayed.

These results are an initial baseline, not a guarantee of real-world performance.

## Run Locally

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

Train and evaluate the model:

```bash
python train.py
```

Predict churn for an example customer:

```bash
python predict.py
```

## Future Improvements
- Compare additional classification algorithms
- Improve the precision-recall trade-off
- Add model evaluation visualizations
- Build an interactive prediction interface
- Deploy the model as an API

## Model Evaluation Visualizations

### Confusion Matrix

The confusion matrix shows how accurately the model identifies customers who stay and customers who churn. At the selected threshold, the model prioritizes catching churners, accepting some false-positive predictions.

![Customer Churn Confusion Matrix](confusion_matrix.png)

### Feature Coefficients

This chart displays the strongest positive and negative Logistic Regression coefficients. Positive coefficients push predictions toward churn, while negative coefficients push them toward staying. These associations do not establish causation.

![Customer Churn Feature Coefficients](feature_importance.png)
