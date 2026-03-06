import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load dataset
data = pd.read_csv(r'C:\Users\Priyanshu\Predictive_pulse\data\clean_patient_data.csv')

# Remove missing values if any
data = data.dropna()

# Features and target
X = data.drop('Stages', axis=1)
y = data['Stages']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

accuracy = {}


# Function to train and evaluate model
def evaluate_model(model, name):

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracy[name] = acc

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Train models
evaluate_model(DecisionTreeClassifier(), "Decision Tree")

evaluate_model(RandomForestClassifier(), "Random Forest")

evaluate_model(SVC(), "SVM")

evaluate_model(KNeighborsClassifier(n_neighbors=5), "KNN")

evaluate_model(RidgeClassifier(), "Ridge Classifier")

evaluate_model(GaussianNB(), "Naive Bayes")


# Accuracy comparison
print("\n==============================")
print("Model Accuracy Comparison")
print("==============================")

for model, acc in accuracy.items():
    print(f"{model} : {acc:.4f}")
    from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def model_selection(X, y):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Ridge Classifier': RidgeClassifier()
    }
    
    model_scores = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        model_scores[name] = np.mean(scores)
        print(f"{name} Average CV Accuracy: {np.mean(scores):.4f}")
    
    best_model_name = max(model_scores, key=model_scores.get)
    print(f"\nBest Model: {best_model_name} with accuracy {model_scores[best_model_name]:.4f}")
    
    
    return models[best_model_name]



best_model = model_selection(X, y)
import pickle

# Train best model
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)

# Save model
pickle.dump(best_model, open("random_forest_model.pkl", "wb"))

print("Model saved successfully!")