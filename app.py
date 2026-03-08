# Predictive Pulse - Blood Pressure Prediction System
# ML Model Training & Deployment: Vikram Yadav
# Flask Integration: Team Project

from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("logistic_model.pkl")

# Stage mapping
stage_map = {
    0: "Normal Blood Pressure",
    1: "Stage 1 Hypertension",
    2: "Stage 2 Hypertension",
    3: "Hypertension Crisis"
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    gender = int(request.form.get("gender", 0))
    age = int(request.form.get("age", 0))
    family = int(request.form.get("family", 0))
    medical = int(request.form.get("medical", 0))
    medicine = int(request.form.get("medicine", 0))
    severity = int(request.form.get("severity", 0))
    breath = int(request.form.get("breath", 0))
    vision = int(request.form.get("vision", 0))
    nose = int(request.form.get("nose", 0))
    time = int(request.form.get("time", 0))
    systolic = int(request.form.get("systolic", 0))
    diastolic = int(request.form.get("diastolic", 0))
    diet = int(request.form.get("diet", 0))

    # Create feature array
    features = np.array([[
        gender,
        age,
        family,
        medical,
        medicine,
        severity,
        breath,
        vision,
        nose,
        time,
        systolic,
        diastolic,
        diet
    ]])

    # Predict
    prediction = int(model.predict(features)[0])

    # Map stage
    result = stage_map.get(prediction, "Unknown")

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)