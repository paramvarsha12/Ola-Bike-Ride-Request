import pandas as pd
import numpy as np

df = pd.read_csv('Ola_Bike_Ride_Request_Dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df['trip_distance'] = haversine(df['pick_lat'], df['pick_lng'], df['drop_lat'], df['drop_lng'])
df['hour'] = df['timestamp'].dt.hour

peak_hour = df['hour'].value_counts().idxmax()
peak_hour_count = df['hour'].value_counts().max()
short_trips_pct = (df[df['trip_distance'] < 5].shape[0] / df.shape[0]) * 100
bangalore_trips = df[(df['pick_lat'].between(12.8, 13.0)) & (df['pick_lng'].between(77.5, 77.7))].shape[0]
total_users = df['user_id'].nunique()
avg_trips_per_user = df.shape[0] / total_users

print("Additional Insights:")
print(f"Peak Hours: {peak_hour}-{'{:02d}'.format(peak_hour+1)} ({peak_hour_count} trips)")
print(f"Short Trips: {short_trips_pct:.2f}% under 5km")
print("Bangalore Hub: Concentrated around 12.9°N, 77.6°E")
print(f"Loyal Users: Average {avg_trips_per_user:.0f} trips per customer")