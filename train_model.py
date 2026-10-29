
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Find and load CSV
BASE_DIR = Path(__file__).resolve().parent

possible_paths = [
    BASE_DIR / "machine_data.csv",
    BASE_DIR.parent / "machine_data.csv"
]

DATA_PATH = next(
    (p for p in possible_paths if p.exists()),
    None
)

if DATA_PATH is None:
    raise FileNotFoundError(
        "machine_data.csv not found. "
        "Place it in the project folder or its parent folder."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Data loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Create target from existing failure columns
failure_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

missing = [c for c in failure_columns if c not in df.columns]

if missing:
    raise ValueError(f"These columns are missing: {missing}")

df["failure_risk"] = (
    df[failure_columns].fillna(0).sum(axis=1) > 0
).astype(int)

# 3. Separate input and output
drop_columns = failure_columns + ["failure_risk"]

for col in ["UDI", "Product ID"]:
    if col in df.columns:
        drop_columns.append(col)

X = df.drop(columns=drop_columns)
y = df["failure_risk"]

# Convert text categories to numeric columns
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.median())

print("\nFeatures prepared successfully!")
print("Failure counts:")
print(y.value_counts())

# 4. Split the data
if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError(
        "Dataset needs at least two classes with "
        "at least two rows in each class."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

print("\nTraining model...")
model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)

print("\nAccuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# 7. Save model and its feature names
model_data = {
    "model": model,
    "features": X.columns.tolist()
}

joblib.dump(
    model_data,
    BASE_DIR / "machine_failure_model.pkl"
)

print("\nSUCCESS! Model saved successfully.")
