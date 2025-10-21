import pandas as pd

df = pd.read_csv('Ola_Bike_Ride_Request_Dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['month'] = df['timestamp'].dt.month
df['day_of_week'] = df['timestamp'].dt.day_name()
df['quarter'] = df['timestamp'].dt.quarter

month_counts = df['month'].value_counts().sort_index()
peak_month = month_counts.idxmax()
peak_month_count = month_counts.max()
day_counts = df['day_of_week'].value_counts()
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
weekends = ['Saturday', 'Sunday']
weekday_total = day_counts[weekdays].sum()
weekend_total = day_counts[weekends].sum()
quarter_counts = df['quarter'].value_counts().sort_index()

print("Temporal Patterns:")
print(f"Monthly peak in month {peak_month} ({peak_month_count:,} trips)")
print(f"Weekend vs weekday variations: {weekend_total:,} weekend trips, {weekday_total:,} weekday trips")
print(f"Seasonal demand fluctuations by quarter: {quarter_counts.to_dict()}")