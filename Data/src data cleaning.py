import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('C:\\Users\\Priyanshu\\Predictive_pulse\\Data\\patient_data .csv')
data.head()


path = ('C:\\Users\\Priyanshu\\Predictive_pulse\\Data\\patient_data .csv')
data.head()
# remove extra spaces from column names
data.columns = data.columns.str.strip()

data = pd.read_csv(path)

print("Dataset Loaded Successfully")
print(data.head())
print(data.shape)
# First 5 rows
print(data.head())

# Check missing values
print(data.isnull().sum())

# Fill missing values with most frequent value
for column in data.columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

# Check again
print(data.isnull().sum())
data.rename(columns={'C': 'Gender'}, inplace=True)
print(data.columns)
#Inconsistency Corrections


# Remove leading and trailing spaces from all string values
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Standardize Yes/No columns
yes_no_cols = ['History', 'Patient', 'TakeMedication', 'BreathShortness',
               'VisualChanges', 'NoseBleeding']

for col in yes_no_cols:
    data[col] = data[col].replace({
        'Yes': 'Yes',
        'No': 'No',
        'yes': 'Yes',
        'no': 'No',
        'YES': 'Yes',
        'NO': 'No'
    })

# Fix Systolic values
data['Systolic'] = data['Systolic'].replace({
    '100+': '100 - 110',
    '121- 130': '121 - 130',
    '111-120': '111 - 120'
})

# Fix Diastolic values
data['Diastolic'] = data['Diastolic'].replace({
    '100+': '100 - 110',
    '130+': '130 - 140'
})

# Fix Stages column inconsistencies
data['Stages'] = data['Stages'].replace({
    'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)',
    'Hypertension (Stage-2)': 'HYPERTENSION (Stage-2)',
    'HYPERTENSIVE CRISIS ': 'HYPERTENSIVE CRISIS'
})

# Check cleaned dataset
print(data.head())

# Check unique values
print("\nUnique values after cleaning:\n")
for col in data.columns:
    print(col, ":", data[col].unique())
    # Check duplicate rows
print(data.duplicated().sum())

# Remove duplicate rows
data.drop_duplicates(inplace=True)
print("Duplicate rows removed. Current shape:", data.shape)

for col in data.columns:
    print(col)
    nominal_features = ['Gender','History','Patient','TakeMedication',
                    'BreathShortness','VisualChanges','NoseBleeding','ControlledDiet']

ordinal_features = [f for f in data.columns if f not in nominal_features]
ordinal_features.remove('Stages')

print(nominal_features)
print(ordinal_features)


for col in nominal_features:
    if set(data[col].unique()) == set(['Yes','No']):
        data[col] = data[col].map({'No':0,'Yes':1})

    elif col == 'Gender':
        data[col] = data[col].map({'Male':0,'Female':1})


data['Age'] = data['Age'].map({'18-34':1,'35-50':2,'51-64':3,'65+':4})

data['Severity'] = data['Severity'].replace({'Mild':0,'Moderate':1,'Severe':2})




data['Systolic'] = data['Systolic'].map({
    '100 - 110':0,
    '111 - 120':1,
    '121 - 130':2,
    '130+':3
})


data['Diastolic'] = data['Diastolic'].map({
    '70 - 80':0,
    '81 - 90':1,
    '91 - 100':2,
    '100+':3
})


data['Stages'] = data['Stages'].map({
    'NORMAL':0,
    'HYPERTENSION (Stage-1)':1,
    'HYPERTENSION (Stage-2)':2,
    'HYPERTENSIVE CRISIS':3
})
print(data.head())


print(ordinal_features)
print(data[ordinal_features].head())
data[ordinal_features] = data[ordinal_features].apply(pd.to_numeric, errors='coerce')
data[ordinal_features] = data[ordinal_features].fillna(0)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

data[ordinal_features] = scaler.fit_transform(data[ordinal_features])
print(data[ordinal_features].head())