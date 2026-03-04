import pandas as pd

print("Starting Data Preparation...\n")

# Load Dataset
file_path ="C:\\Users\\Priyanshu\\Desktop\\Predictive Pulse\\Data\\patient_data .csv"

df = pd.read_csv(file_path)

print("Dataset Loaded Successfully!\n")
# Display first rows
print("First 5 rows:")
print(df.head())

# Dataset Info
print("\nDataset Information:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())
print("\nChecking Missing Values:")

missing_values = df.isnull().sum()

print(missing_values)
# Rename column C to Gender
df.rename(columns={'C': 'Gender'}, inplace=True)

print("\nColumn renamed successfully")
print("\nChecking duplicates...")

print("Duplicate rows:", df.duplicated().sum())
print("\nRemoving duplicates...")

before = df.shape[0]

df = df.drop_duplicates()

after = df.shape[0]

print("Duplicates removed:", before - after)
print("\nFinal dataset shape:")
print(df.shape)
df.to_csv("Data/clean_patient_data.csv", index=False)

print("\nClean dataset saved successfully")
