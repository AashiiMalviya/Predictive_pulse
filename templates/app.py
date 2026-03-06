from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open(r"C:\Users\Priyanshu\Predictive_pulse\model\random_forest_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    features = np.array([features])

    prediction = model.predict(features)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Hypertension Stage: {prediction[0]}"
    )

if __name__ == "__main__":
    app.run(debug=True)