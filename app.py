from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Random Forest Model
model = pickle.load(open("random_forest_model.pkl", "rb"))

# Stage Mapping
stage_map = {
    0: "NORMAL",
    1: "HYPERTENSION (Stage-1)",
    2: "HYPERTENSION (Stage-2)",
    3: "HYPERTENSIVE CRISIS"
}

# Color Mapping
color_map = {
    0: "#10B981",
    1: "#F59E0B",
    2: "#F97316",
    3: "#EF4444"
}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["Age"])
    systolic = int(request.form["Systolic"])
    diastolic = int(request.form["Diastolic"])

    input_data = np.array([[age, systolic, diastolic]])

    prediction = model.predict(input_data)[0]
    confidence = max(model.predict_proba(input_data)[0]) * 100

    result = stage_map[prediction]
    color = color_map[prediction]

    return render_template(
        "index.html",
        prediction_text=result,
        result_color=color,
        confidence=round(confidence,2)
    )


if __name__ == "__main__":
    app.run(debug=True)