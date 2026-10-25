import pandas as pd

df = pd.read_csv("churn.csv")

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nColumn information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nChurn distribution:")
print(df["Churn"].value_counts())


print("\nTotalCharges blank values:")
print((df["TotalCharges"].str.strip() == "").sum())

print("\nTotalCharges sample values:")
print(df["TotalCharges"].head(10).tolist())


# Convert blank TotalCharges values to missing values
df["TotalCharges"] = df["TotalCharges"].str.strip()
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Check missing values after conversion
print("\nMissing TotalCharges:", df["TotalCharges"].isnull().sum())

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

print("Shape after cleaning:", df.shape)
print("TotalCharges data type:", df["TotalCharges"].dtype)
