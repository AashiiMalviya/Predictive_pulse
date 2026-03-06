import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv("Data/clean_patient_data.csv")


X = data.drop('target_column', axis=1)  # 
y = data['target_column']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)