# Predictive Pulse - Blood Pressure Prediction System
# Model Training Script
# Developed by: Vikram Yadav

import pandas as pd
import joblib

# ==============================
# Load Dataset
# ==============================

data = pd.read_csv(
    r"C:\Users\Priyanshu\Predictive_pulse\data\clean_patient_data.csv"
)

# Remove missing values
data = data.dropna()

print(data.head())

# ==============================
# Features and Target
# ==============================

X = data.drop("Stages", axis=1)
y = data["Stages"]

# ==============================
# Train Test Split
# ==============================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Import Models
# ==============================

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

accuracy = {}

# ==============================
# Logistic Regression
# ==============================

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\nLogistic Regression Accuracy:", acc)

accuracy["Logistic Regression"] = acc

# ==============================
# Decision Tree
# ==============================

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Decision Tree Accuracy:", acc)

accuracy["Decision Tree"] = acc

# ==============================
# Random Forest
# ==============================

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Random Forest Accuracy:", acc)

accuracy["Random Forest"] = acc

# ==============================
# SVM
# ==============================

svm = SVC()
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("SVM Accuracy:", acc)

accuracy["SVM"] = acc

# ==============================
# KNN
# ==============================

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("KNN Accuracy:", acc)

accuracy["KNN"] = acc

# ==============================
# Ridge Classifier
# ==============================

rc = RidgeClassifier()
rc.fit(X_train, y_train)

y_pred = rc.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("RidgeClassifier Accuracy:", acc)

accuracy["RidgeClassifier"] = acc

# ==============================
# Naive Bayes
# ==============================

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Naive Bayes Accuracy:", acc)

accuracy["Naive Bayes"] = acc

# ==============================
# Model Accuracy Comparison
# ==============================

print("\nModel Accuracy Comparison")

for model, acc in accuracy.items():
    print(model, ":", acc)

# ==============================
# Force Best Model = Logistic Regression
# ==============================

best_model = logreg

print("\nBest Model Selected: Logistic Regression")

# ==============================
# Save Model
# ==============================

joblib.dump(best_model, "logistic_model.pkl")

print("Model saved successfully as logistic_model.pkl")