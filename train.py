
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# 1. Load and clean data
df = pd.read_csv("churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce",
)
df = df.dropna(subset=["TotalCharges"])

# 2. Separate features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 3. Stratified train / validation / test split
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev,
)

# 4. Preprocessing
numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 5. Train selected model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
    )),
])

model.fit(X_train, y_train)

# 6. Confirm selected threshold on validation data
threshold = 0.30
val_prob = model.predict_proba(X_val)[:, 1]
val_pred = (val_prob >= threshold).astype(int)

print("VALIDATION RESULTS")
print("Precision:", round(precision_score(y_val, val_pred, zero_division=0), 3))
print("Recall:", round(recall_score(y_val, val_pred, zero_division=0), 3))
print("F1-score:", round(f1_score(y_val, val_pred, zero_division=0), 3))

# 7. Final evaluation on the held-out test set
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype(int)

print("\nFINAL TEST RESULTS")
print("Precision:", round(precision_score(y_test, test_pred, zero_division=0), 3))
print("Recall:", round(recall_score(y_test, test_pred, zero_division=0), 3))
print("F1-score:", round(f1_score(y_test, test_pred, zero_division=0), 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nClassification report:")
print(classification_report(
    y_test,
    test_pred,
    target_names=["Stayed", "Churned"],
    zero_division=0,
))

# 8. Save the complete pipeline
joblib.dump(
    {"model": model, "threshold": threshold},
    "churn_model.joblib",
)

print("\nSaved model to churn_model.joblib")
