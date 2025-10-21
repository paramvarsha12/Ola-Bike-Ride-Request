import pandas as pd

df = pd.read_csv('Ola_Bike_Ride_Request_Dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Top 10 popular routes (by pickup/drop coordinates)
top_routes = (
    df.groupby(['pick_lat', 'pick_lng', 'drop_lat', 'drop_lng'])
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print("Operational Insights:")
print("Top 10 popular routes identified (pickup/drop coordinates and trip counts):")
print(top_routes)

# Driver allocation recommendations
# Find hours and locations with highest demand
df['hour'] = df['timestamp'].dt.hour
hourly_demand = df['hour'].value_counts().sort_index()
peak_hours = hourly_demand[hourly_demand == hourly_demand.max()].index.tolist()
print(f"\nDriver allocation recommendation: Allocate more drivers during peak hours {peak_hours}.")

# Pricing optimization opportunities
# Example: Identify hours with high demand for dynamic pricing
high_demand_hours = hourly_demand[hourly_demand > hourly_demand.mean()].index.tolist()
print(f"Pricing optimization: Consider dynamic pricing during high demand hours {high_demand_hours}.")