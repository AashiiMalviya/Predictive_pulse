import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


print("Loading Dataset...")

df = pd.read_csv("Data/clean_patient_data.csv")

target = "Stages"

X = df.drop(target, axis=1)
y = df[target]

encoders = {}

# Encode ONLY categorical columns
cat_cols = X.select_dtypes(include=['object']).columns

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Model...")

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Model Trained Successfully")

data = {
    "model": model,
    "encoders": encoders,
    "target_encoder": target_encoder
}

with open("src/bp_model.pkl", "wb") as f:
    pickle.dump(data, f)

print("Model Saved")


pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("Model Accuracy:", acc)

cm = confusion_matrix(y_test, pred)
print(cm)


importance = model.feature_importances_

for col, score in zip(X.columns, importance):
    print(col, ":", score)