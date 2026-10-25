
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained pipeline
saved = joblib.load("churn_model.joblib")
model = saved["model"]

# Get feature names after preprocessing
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
coefficients = classifier.coef_[0]

# Build a table of model coefficients
importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
})

# Select the strongest positive and negative coefficients
importance = importance.sort_values("Coefficient")
selected = pd.concat([
    importance.head(10),
    importance.tail(10),
]).sort_values("Coefficient")

# Plot and save
plt.figure(figsize=(10, 8))
plt.barh(selected["Feature"], selected["Coefficient"])
plt.axvline(0, linewidth=1)
plt.xlabel("Logistic Regression Coefficient")
plt.title("Features Associated with Churn Predictions")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

print("Saved feature_importance.png")
print("\nStrongest positive coefficients:")
print(importance.tail(10).to_string(index=False))
