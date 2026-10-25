
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

# Load and clean the dataset
df = pd.read_csv("churn.csv")
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(), errors="coerce"
)
df = df.dropna(subset=["TotalCharges"])

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Recreate the same test split
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev
)

# Load the trained model
saved = joblib.load("churn_model.joblib")
model = saved["model"]
threshold = saved["threshold"]

# Predict on the test set
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

# Save confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Stayed", "Churned"],
    cmap="Blues",
)
plt.title("Customer Churn - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

print("Saved confusion_matrix.png")
