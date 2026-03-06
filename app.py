from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
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
        return render_template("index.html", prediction_text="Error: Model file not found")

    try:

        features = [[

        1 if request.form.get("Gender")=="Female" else 0,

        float(request.form.get("Age")),

        1 if request.form.get("History")=="on" else 0,

        1 if request.form.get("Patient")=="on" else 0,

        1 if request.form.get("TakeMedication")=="on" else 0,

        float(request.form.get("Severity",0.5)),

        1 if request.form.get("BreathShortness")=="on" else 0,

        1 if request.form.get("VisualChanges")=="on" else 0,

        1 if request.form.get("NoseBleeding")=="on" else 0,

        float(request.form.get("WhenDiagnosed",0.2)),

        float(request.form.get("Systolic")),

        float(request.form.get("Diastolic")),

        1 if request.form.get("ControlledDiet")=="on" else 0

        ]]

        prediction = model.predict(features)[0]

        stages = {
        0:"Normal",
        1:"Elevated",
        2:"Stage 1 Hypertension",
        3:"Stage 2 Hypertension"
        }

        result = stages.get(prediction,"Unknown")

        return render_template(
        "index.html",
        prediction_text=f"Prediction: {result}"
        )

    except Exception as e:
        return render_template(
        "index.html",
        prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)