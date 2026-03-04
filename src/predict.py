import pickle
import pandas as pd

print("Loading Model...")

with open("model/bp_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
encoders = data["encoders"]
target_encoder = data["target_encoder"]

print("Model Loaded Successfully")

# Example patient input
patient = {
    "Gender": "Male",
    "Age": "35-50",
    "History": "Yes",
    "Patient": "No",
    "TakeMedication": "No",
    "Severity": "Mild",
    "BreathShortness": "No",
    "VisualChanges": "No",
    "NoseBleeding": "No",
    "Whendiagnoused": "<1 Year",
    "Systolic": "111 - 120",
    "Diastolic": "81 - 90",
    "ControlledDiet": "No"
}

df = pd.DataFrame([patient])

# Encode input
for col in df.columns:
    le = encoders[col]
    df[col] = le.transform(df[col])

# Prediction
prediction = model.predict(df)

stage = target_encoder.inverse_transform(prediction)

print("Predicted BP Stage:", stage[0])