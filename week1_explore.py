import pandas as pd

data = pd.read_csv("fake_news.csv")

print("Shape of dataset (rows, columns):")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nFirst 5 rows:")
print(data.head())

print("\nLabel distribution (how many real vs fake):")
print(data['label'].value_counts())

print("\nMissing values in each column:")
print(data.isnull().sum())