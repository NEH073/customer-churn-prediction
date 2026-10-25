
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# CPU training is sufficient for this dataset
device = torch.device("cpu")
THRESHOLD = 0.30
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001

# 1. Load and clean data
df = pd.read_csv("churn.csv")
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce",
)
df = df.dropna(subset=["TotalCharges"])

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

# 2. Reproduce the existing train/validation/test split
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev
)

# 3. Fit preprocessing on training data only
numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
X_val_processed = preprocessor.transform(X_val).astype(np.float32)

# Convert sparse output if the encoder produces it
if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()
if hasattr(X_val_processed, "toarray"):
    X_val_processed = X_val_processed.toarray()

y_train_np = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
y_val_np = y_val.to_numpy(dtype=np.float32)

# 4. Address class imbalance using training-set class weights
positive_count = float(y_train.sum())
negative_count = float(len(y_train) - y_train.sum())
pos_weight = torch.tensor(
    [negative_count / positive_count], dtype=torch.float32
)

train_dataset = TensorDataset(
    torch.from_numpy(X_train_processed),
    torch.from_numpy(y_train_np),
)
generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)

# 5. Small feed-forward neural network
class ChurnNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)

model = ChurnNetwork(X_train_processed.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

# 6. Train using training data; monitor validation F1
best_f1 = -1.0
best_state = None
best_epoch = 0
patience = 7
epochs_without_improvement = 0

X_val_tensor = torch.from_numpy(X_val_processed).to(device)

for epoch in range(EPOCHS):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(X_val_tensor)).cpu().numpy().ravel()

    val_pred = (val_prob >= THRESHOLD).astype(int)
    val_f1 = f1_score(y_val_np, val_pred, zero_division=0)

    print(f"Epoch {epoch + 1:02d}: validation F1={val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch + 1
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping.")
        break

# Keep the best validation checkpoint
model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    val_prob = torch.sigmoid(model(X_val_tensor)).cpu().numpy().ravel()

val_pred = (val_prob >= THRESHOLD).astype(int)

print("\nBEST PYTORCH VALIDATION RESULTS")
print("Best epoch:", best_epoch)
print("Threshold:", THRESHOLD)
print("Precision:", round(precision_score(y_val_np, val_pred, zero_division=0), 3))
print("Recall:", round(recall_score(y_val_np, val_pred, zero_division=0), 3))
print("F1-score:", round(f1_score(y_val_np, val_pred, zero_division=0), 3))

print("\nValidation complete. Test set has not been evaluated.")
