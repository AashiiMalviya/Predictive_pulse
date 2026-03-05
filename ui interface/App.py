import streamlit as st
import pickle
import pandas as pd

# Page Config
st.set_page_config(page_title="Predictive Pulse", layout="centered")

st.title("Predictive Pulse")
st.subheader("AI Based Blood Pressure Prediction System")

st.write("Enter patient details to predict BP category")

# Load Model
data = pickle.load(open("../src/bp_model.pkl", "rb"))

model = data["model"]
encoders = data["encoders"]
target_encoder = data["target_encoder"]

# Patient Details
st.markdown("### Patient Information")

name = st.text_input("Patient Name")
age = st.number_input("Age", min_value=1, max_value=100)
gender = st.selectbox("Gender", ["Male", "Female"])

# Health Data
st.markdown("### Health Parameters")

weight = st.number_input("Weight (kg)")
height = st.number_input("Height (cm)")
heart_rate = st.number_input("Heart Rate (bpm)")

smoking = st.selectbox("Smoking Habit", ["Yes", "No"])
exercise = st.selectbox("Regular Exercise", ["Yes", "No"])

# Predict Button
if st.button("Predict Blood Pressure Category"):

    input_data = pd.DataFrame({
        "Age":[age],
        "Gender":[gender],
        "Weight":[weight],
        "Height":[height],
        "Heart_Rate":[heart_rate],
        "Smoking":[smoking],
        "Exercise":[exercise]
    })

    # Encoding
    for col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

    prediction = model.predict(input_data)
    result = target_encoder.inverse_transform(prediction)

    st.success(f"Patient: {name}")
    st.success(f"Predicted BP Category: {result[0]}")