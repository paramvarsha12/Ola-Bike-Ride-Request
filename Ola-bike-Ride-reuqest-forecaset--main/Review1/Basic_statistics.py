import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('Ola_Bike_Ride_Request_Dataset.csv')

# Trip Distance Calculation (Haversine formula)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df['trip_distance'] = haversine(df['pick_lat'], df['pick_lng'], df['drop_lat'], df['drop_lng'])

# Trip Distance Stats
mean_dist = df['trip_distance'].mean()
median_dist = df['trip_distance'].median()
mode_dist = df['trip_distance'].mode()[0]
min_dist = df['trip_distance'].min()
max_dist = df['trip_distance'].max()
range_dist = max_dist - min_dist
std_dist = df['trip_distance'].std()

print("Trip Distance (km):")
print(f"Mean: {mean_dist:.2f}")
print(f"Median: {median_dist:.2f}")
print(f"Mode: {mode_dist:.2f}")
print(f"Min: {min_dist:.2f}")
print(f"Max: {max_dist:.2f}")
print(f"Range: {range_dist:.2f}")
print(f"Standard Deviation: {std_dist:.2f}")

# Geographic Coverage
lat_min, lat_max = df['pick_lat'].min(), df['pick_lat'].max()
lng_min, lng_max = df['pick_lng'].min(), df['pick_lng'].max()
print("\nGeographic Coverage:")
print(f"Pickup Latitude Range: {lat_min:.2f} to {lat_max:.2f}")
print(f"Pickup Longitude Range: {lng_min:.2f} to {lng_max:.2f}")

# Primary Service Area (Bangalore)
bangalore_trips = df[(df['pick_lat'].between(12.8, 13.0)) & (df['pick_lng'].between(77.5, 77.7))].shape[0]
print(f"Primary Service Area: Bangalore Metropolitan Region (Trips: {bangalore_trips})")

# User Statistics
total_users = df['user_id'].nunique()
avg_trips_per_user = df.shape[0] / total_users
most_active_user_trips = df['user_id'].value_counts().max()
print("\nUser Statistics:")
print(f"Total Unique Users: {total_users}")
print(f"Average Trips per User: {avg_trips_per_user:.2f}")
print(f"Most Active User: {most_active_user_trips} trips")