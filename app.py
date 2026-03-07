
# Predictive Pulse - Blood Pressure Prediction System
# ML Model Training & Deployment: Vikram Yadav
# Flask Integration: Team Project
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("logreg_model.pkl")

# Hypertension stage mapping
stage_map = {
    0: ("Normal Blood Pressure", "LOW RISK"),
    1: ("Stage 1 Hypertension", "MODERATE RISK"),
    2: ("Stage 2 Hypertension", "HIGH RISK")
}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        family_history = int(request.form["family_history"])
        medication = int(request.form["medication"])
        severity = int(request.form["severity"])
        breath = int(request.form["breath"])
        vision = int(request.form["vision"])
        nosebleed = int(request.form["nosebleed"])
        diagnosis_time = int(request.form["diagnosis_time"])
        systolic = int(request.form["systolic"])
        diastolic = int(request.form["diastolic"])
        diet = int(request.form["diet"])
        medical_care = int(request.form["medical_care"])

        features = np.array([[

            age,
            gender,
            family_history,
            medication,
            severity,
            breath,
            vision,
            nosebleed,
            diagnosis_time,
            systolic,
            diastolic,
            diet,
            medical_care

        ]])

        prediction = model.predict(features)[0]

        result, risk = stage_map[prediction]

        return render_template(
            "index.html",
            prediction_text=result,
            risk=risk
        )

    except:
        return render_template(
            "index.html",
            prediction_text="Error in Prediction"
        )


if __name__ == "__main__":
    app.run(debug=True)