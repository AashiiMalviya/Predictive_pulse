from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

# Load saved model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Hypertension Prediction App Running Successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([data])
    return "Prediction: " + str(prediction[0])

if __name__ == "__main__":
    app.run(debug=True)