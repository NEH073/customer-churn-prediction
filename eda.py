
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("churn.csv")

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)
df = df.dropna(subset=["TotalCharges"])

# Compare tenure for customers who stayed vs left
print(df.groupby("Churn")["tenure"].describe())

# Visualize tenure distribution
df.boxplot(column="tenure", by="Churn")
plt.title("Customer Tenure vs Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Tenure (months)")
plt.show()



print("\nChurn rate by contract type:")

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print(contract_churn.round(2))
