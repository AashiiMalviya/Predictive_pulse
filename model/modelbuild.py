import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load dataset
data = pd.read_csv(r'C:\Users\Priyanshu\Predictive_pulse\data\clean_patient_data.csv')

# Remove missing values
data = data.dropna()

# Features and target
X = data.drop('Stages', axis=1)
y = data['Stages']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
logreg = LogisticRegression(max_iter=1000)

# Train model
logreg.fit(X_train, y_train)

# Predictions
y_pred = logreg.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Logistic Regression Model")
print("==============================")

print("Accuracy:", acc)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# Save model
joblib.dump(logreg, "logreg_model.pkl")

print("✅ Model saved as logreg_model.pkl")