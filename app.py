from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load ML model
model = joblib.load("logreg_model.pkl")

# Stage Mapping
stage_map = {
    0: ("Normal Blood Pressure", "LOW RISK"),
    1: ("Stage 1 Hypertension", "MODERATE RISK"),
    2: ("Stage 2 Hypertension", "HIGH RISK"),
    3: ("Hypertensive Crisis", "EMERGENCY")
}

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():

    try:
        features = [
            int(request.form["gender"]),
            int(request.form["age"]),
            int(request.form["family_history"]),
            int(request.form["medical_care"]),
            int(request.form["bp_med"]),
            int(request.form["symptom"]),
            int(request.form["breath"]),
            int(request.form["vision"]),
            int(request.form["nosebleed"]),
            int(request.form["diagnosis"]),
            int(request.form["systolic"]),
            int(request.form["diastolic"]),
            int(request.form["diet"])
        ]

        final = np.array([features])

        prediction = model.predict(final)[0]

        stage, risk = stage_map[prediction]

        return render_template(
            "index.html",
            prediction_text=stage,
            risk=risk
        )

    except:
        return render_template(
            "index.html",
            prediction_text="Error in Prediction",
            risk="Check Inputs"
        )


if __name__ == "__main__":
    app.run(debug=True)