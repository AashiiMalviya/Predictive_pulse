from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
try:
    model = joblib.load("logreg_model.pkl")
except:
    model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return render_template("index.html", prediction_text="Model file not found")

    try:
        gender = 1 if request.form.get("Gender") == "Female" else 0
        age = float(request.form.get("Age"))
        systolic = float(request.form.get("Systolic"))
        diastolic = float(request.form.get("Diastolic"))

        history = 1 if request.form.get("History") == "on" else 0
        breath = 1 if request.form.get("BreathShortness") == "on" else 0

        features = np.array([[gender, age, systolic, diastolic, history, breath]])

        prediction = model.predict(features)[0]

        stages = {
            0: "Normal Blood Pressure",
            1: "Stage 1 Hypertension",
            2: "Stage 2 Hypertension",
            3: "Hypertensive Crisis"
        }

        result = stages.get(prediction, "Unknown")

        return render_template(
            "index.html",
            prediction_text=f"Prediction Result: {result}"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)