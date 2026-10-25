import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# 1. Load and clean data
df = pd.read_csv("churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 2. Use the same train/validation/test split as train.py
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev
)

# 3. Preprocess features
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Fit preprocessing ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# 4. Train both models
models = {
    "Logistic Regression": LogisticRegression(
        C=1, max_iter=1000, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train_processed, y_train)
    val_probabilities = model.predict_proba(X_val_processed)[:, 1]
    val_predictions = (val_probabilities >= 0.30).astype(int)

    precision = precision_score(y_val, val_predictions, zero_division=0)
    recall = recall_score(y_val, val_predictions, zero_division=0)
    f1 = f1_score(y_val, val_predictions, zero_division=0)

    results[name] = {
        "model": model,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print(f"\n{name} — VALIDATION RESULTS")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

# 5. Select using validation F1; use recall to break ties
best_name = max(
    results,
    key=lambda name: (results[name]["f1"], results[name]["recall"])
)

# Keep preprocessing and the selected classifier together
best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", results[best_name]["model"])
])

# Fit the complete pipeline on training data
best_model.fit(X_train, y_train)

joblib.dump(
    {"model": best_model, "threshold": 0.30},
    "churn_model.joblib"
)

print(f"\nSelected model: {best_name}")
print("Saved preprocessing + model pipeline to churn_model.joblib")
joblib.dump(
    {"model": best_model, "threshold": 0.30},
    "churn_model.joblib"
)
print("Saved selected model to churn_model.joblib")

print(f"\nSelected model based on validation results: {best_name}")

# 6. Evaluate the selected model on the test set once
X_test_processed = preprocessor.transform(X_test)
test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= 0.30).astype(int)

print("\nSELECTED MODEL — FINAL TEST RESULTS")
print(f"Model:     {best_name}")
print(f"Precision: {precision_score(y_test, test_predictions, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, test_predictions, zero_division=0):.3f}")
print(f"F1-score:  {f1_score(y_test, test_predictions, zero_division=0):.3f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["Stayed", "Churned"],
    zero_division=0
))
