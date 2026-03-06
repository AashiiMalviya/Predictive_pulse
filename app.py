from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("logreg_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = [

    1 if request.form.get("Gender")=="Female" else 0,
    float(request.form.get("Age")),

    1 if request.form.get("History")=="on" else 0,
    1 if request.form.get("Patient")=="on" else 0,
    1 if request.form.get("TakeMedication")=="on" else 0,

    float(request.form.get("Severity") or 0),

    1 if request.form.get("BreathShortness")=="on" else 0,
    1 if request.form.get("VisualChanges")=="on" else 0,
    1 if request.form.get("NoseBleeding")=="on" else 0,

    float(request.form.get("WhenDiagnosed") or 0),

    float(request.form.get("Systolic")),
    float(request.form.get("Diastolic")),

    1 if request.form.get("ControlledDiet")=="on" else 0
    ]

    prediction = model.predict([features])[0]

    stages={
    0:"Normal",
    1:"Stage 1 Hypertension",
    2:"Stage 2 Hypertension",
    3:"Hypertensive Crisis"
    }

    result=stages.get(prediction,"Unknown")

    return render_template("index.html",prediction_text=f"Prediction: {result}")

if __name__=="__main__":
    app.run(debug=True)