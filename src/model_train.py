import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Starting Model Training...\n")

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_dir, ".."))

data_path = os.path.join(project_root, "Data", "patient_data.csv")
model_dir = os.path.join(project_root, "model")
model_path = os.path.join(model_dir, "bp_model.pkl")

print("Loading dataset from:", data_path)

df = pd.read_csv(data_path)

print("Dataset Loaded:", df.shape)

X = df.drop(columns=["Systolic_BP", "Diastolic_BP"])
y = df[["Systolic_BP", "Diastolic_BP"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data Split Completed")

model = RandomForestRegressor()

print("Training Model...")
model.fit(X_train, y_train)

print("Model Training Completed")

os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, model_path)

print("Model Saved Successfully!")