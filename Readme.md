# 🫀 Predictive Pulse – Blood Pressure Prediction System

## 📌 Project Overview

Predictive Pulse is a Machine Learning based web application that predicts a patient's **Blood Pressure Category** using health-related features.
The system analyzes patient information and classifies the blood pressure stage such as:

* Normal
* Hypertension (Stage-1)
* Hypertension (Stage-2)
* Hypertensive Crisis

The model is trained using Machine Learning techniques and deployed using **Flask**.

---

## 🎯 Objective

The goal of this project is to build an intelligent system that can help in **early detection of hypertension risk** using patient health data.

---

## 🧠 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Handling Missing Values
4. Removing Duplicate Records
5. Categorical Data Encoding
6. Feature Scaling (MinMaxScaler)
7. Train-Test Split
8. Model Training (Logistic Regression)
9. Model Evaluation
10. Model Deployment with Flask

---

## 📊 Features Used in Model

* Age
* Gender
* History
* Patient
* Take Medication
* Breath Shortness
* Visual Changes
* Nose Bleeding
* Controlled Diet
* Severity
* When Diagnosed
* Systolic Blood Pressure
* Diastolic Blood Pressure

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Flask
* HTML / CSS

---

## 📂 Project Structure

Predictive_pulse

Data
└ patient_data.csv

src
└ data_cleaning.py

model
└ logreg_model.pkl

templates
└ index.html

app.py
README.md

---

## ⚙️ Installation

Clone the repository

git clone https://github.com/your-username/Predictive_pulse.git

Move to project folder

cd Predictive_pulse

Install dependencies

pip install -r requirements.txt

Run the Flask app

python app.py

---

## 📈 Model Performance

The Logistic Regression model is trained on the processed dataset and evaluated using accuracy metrics.

---

## 🚀 Future Improvements

* Add more medical features
* Improve model accuracy using advanced algorithms
* Deploy the application online
* Add patient dashboard

---

## 👨‍💻 Author

Priyanshu Tiwari
B.Tech AI / ML Engineer
Vikram Yadav 
B. Tech (CSE)