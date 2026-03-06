from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your model (ensure logreg_model.pkl is in the same folder)
try:
    model = joblib.load("logreg_model.pkl")
except:
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', prediction_text="Error: Model file not found.")
    
    # Get data from form
    try:
        # Map form data to match your model's expected features
        # Gender (Male: 0, Female: 1), Age, History, Patient, TakeMedication, etc.
        features = [
            1 if request.form.get('Gender') == 'Female' else 0,
            float(request.form.get('Age')) / 100, # Example normalization
            1 if request.form.get('History') == 'on' else 0,
            1 if request.form.get('Patient') == 'on' else 0,
            1 if request.form.get('TakeMedication') == 'on' else 0,
            0.5, # Severity default
            1 if request.form.get('BreathShortness') == 'on' else 0,
            1 if request.form.get('VisualChanges') == 'on' else 0,
            1 if request.form.get('NoseBleeding') == 'on' else 0,
            0.2, # Whendiagnoused default
            float(request.form.get('Systolic')) / 200,
            float(request.form.get('Diastolic')) / 130,
            1 if request.form.get('ControlledDiet') == 'on' else 0
        ]
        
        prediction = model.predict([features])[0]
        stages = {1: "Stage 1: Elevated", 2: "Stage 2: Hypertension", 3: "Stage 3: Crisis"}
        result_text = stages.get(prediction, "Unknown Stage")
        
        return render_template('index.html', 
                               prediction_text=f"Predicted: {result_text}",
                               result_color="red" if prediction > 1 else "green")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, port=5000)