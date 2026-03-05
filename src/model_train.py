import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Loading Dataset...")

# Load dataset
df = pd.read_csv("Data/clean_patient_data.csv")

print("Dataset Loaded Successfully")

# Target column
target = "Stages"

X = df.drop(target, axis=1)
y = df[target]


# Encode categorical data
encoders = {}

for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Model...")

# Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Trained Successfully")

# Save model + encoders
data = {
    "model": model,
    "encoders": encoders,
    "target_encoder": target_encoder
}

with open("src/bp_model.py", "wb") as f:
    pickle.dump(data, f)

print("Model Saved -> src/bp_model.py")