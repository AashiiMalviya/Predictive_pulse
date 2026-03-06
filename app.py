
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/random_forest_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    systolic = float(request.form["systolic"])
    non_systolic = float(request.form["non_systolic"])
    anemia = float(request.form["anemia"])
    non_anemia = float(request.form["non_anemia"])

    data = [[systolic, non_systolic, anemia, non_anemia]]

    columns = ["systolic","non_systolic","anemia","non_anemia"]

    df = pd.DataFrame(data, columns=columns)

    prediction = model.predict(df)


    return render_template(
        "index.html",
        prediction_text=f"Predicted Hypertension Stage : {prediction[0]}"
    )

if __name__ == "__main__":
    app.run(debug=True)