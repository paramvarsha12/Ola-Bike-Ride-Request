import pandas as pd

df = pd.read_csv('Ola_Bike_Ride_Request_Dataset.csv')

print("Number of Rows & Columns:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nData Types:")
print(df.dtypes)
print("\nSample Data:")
print(df.head())