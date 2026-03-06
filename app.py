from flask import Flask, render_template, request
import joblib

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
        return render_template("index.html", prediction_text="Model not found")

    try:

        gender = 1 if request.form.get("Gender") == "Female" else 0
        age = float(request.form.get("Age", 0))

        history = 1 if request.form.get("History") == "on" else 0
        patient = 1 if request.form.get("Patient") == "on" else 0
        medication = 1 if request.form.get("TakeMedication") == "on" else 0

        severity = float(request.form.get("Severity", 0.5))

        breath = 1 if request.form.get("BreathShortness") == "on" else 0
        visual = 1 if request.form.get("VisualChanges") == "on" else 0
        nose = 1 if request.form.get("NoseBleeding") == "on" else 0

        diagnosed = float(request.form.get("WhenDiagnosed", 0.2))

        systolic = float(request.form.get("Systolic", 0))
        diastolic = float(request.form.get("Diastolic", 0))

        diet = 1 if request.form.get("ControlledDiet") == "on" else 0

        features = [[
            gender,
            age,
            history,
            patient,
            medication,
            severity,
            breath,
            visual,
            nose,
            diagnosed,
            systolic,
            diastolic,
            diet
        ]]

        prediction = model.predict(features)[0]

        stages = {
            0: "Normal",
            1: "Elevated",
            2: "Stage 1 Hypertension",
            3: "Stage 2 Hypertension"
        }

        result = stages.get(prediction, "Unknown")

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