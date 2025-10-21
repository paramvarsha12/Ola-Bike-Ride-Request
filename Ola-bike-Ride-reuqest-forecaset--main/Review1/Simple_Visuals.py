import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth in kilometers"""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

print("=== CREATING SIMPLE VISUALIZATIONS ===")

# Load the actual dataset
df = pd.read_csv('ct_rr.csv')
print(f"Loaded {df.shape[0]:,} records")

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['ts'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()

# Calculate trip distances
print("Calculating trip distances...")
df['trip_distance'] = haversine(df['pick_lat'], df['pick_lng'], df['drop_lat'], df['drop_lng'])

# 1. Geographic Service Coverage
print("Creating geographic service coverage...")
fig, ax = plt.subplots(figsize=(10, 8))

# Sample data for performance
sample_df = df.iloc[::5000].copy()  # Take every 5000th point

# Calculate density for coloring
coords = np.vstack([sample_df['pick_lng'], sample_df['pick_lat']])
density = gaussian_kde(coords)(coords)

# Create density categories
low_density = density < np.percentile(density, 33)
medium_density = (density >= np.percentile(density, 33)) & (density < np.percentile(density, 67))
high_density = density >= np.percentile(density, 67)

# Plot with density-based colors
ax.scatter(sample_df.loc[low_density, 'pick_lng'], sample_df.loc[low_density, 'pick_lat'], 
           c='lightblue', s=2, alpha=0.6, label='Low Density')
ax.scatter(sample_df.loc[medium_density, 'pick_lng'], sample_df.loc[medium_density, 'pick_lat'], 
           c='green', s=2, alpha=0.6, label='Medium Density')
ax.scatter(sample_df.loc[high_density, 'pick_lng'], sample_df.loc[high_density, 'pick_lat'], 
           c='red', s=2, alpha=0.6, label='High Density')

# Add Bangalore center marker
bangalore_lat, bangalore_lng = 12.9716, 77.5946
ax.scatter(bangalore_lng, bangalore_lat, marker='*', s=300, c='red', 
          edgecolors='black', linewidth=2, label='Bangalore Center')

ax.set_title('Geographic Service Coverage', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend()

plt.savefig('simple_geographic_coverage.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Daily Demand Distribution
print("Creating daily demand distribution...")
fig, ax = plt.subplots(figsize=(8, 8))

day_counts = df['day_of_week'].value_counts()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = day_counts.reindex(days_order)

# Colors
colors = ['darkblue', 'green', 'purple', 'orange', 'red', 'lightblue', 'lavender']
wedges, texts, autotexts = ax.pie(day_counts.values, labels=day_counts.index, 
                                  autopct='%1.0f%%', startangle=90, colors=colors)

ax.set_title('Daily Demand Distribution', fontsize=14, fontweight='bold')

plt.savefig('simple_daily_demand.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Hourly Demand Pattern
print("Creating hourly demand pattern...")
fig, ax = plt.subplots(figsize=(12, 6))

hour_counts = df['hour'].value_counts().sort_index()

# Define colors based on peak hours (0-2 AM)
colors = ['red' if hour in [0, 1, 2] else 'green' for hour in hour_counts.index]

bars = ax.bar(hour_counts.index, hour_counts.values, color=colors, alpha=0.7)

ax.set_title('Hourly Demand Pattern', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Number of trips', fontsize=12)
ax.set_xticks(range(0, 24, 2))
ax.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', alpha=0.7, label='Peak Hours (0-2 AM)'),
                   Patch(facecolor='green', alpha=0.7, label='Regular Hours')]
ax.legend(handles=legend_elements)

plt.savefig('simple_hourly_demand.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Trip Distance Distribution
print("Creating trip distance distribution...")
fig, ax = plt.subplots(figsize=(8, 6))

# Categorize distances
df['distance_category'] = pd.cut(df['trip_distance'], 
                                bins=[0, 5, 10, 15, float('inf')], 
                                labels=['0-5km', '5-10km', '10-15km', '15+km'])

distance_counts = df['distance_category'].value_counts().sort_index()

# Create bar chart with colors
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax.bar(distance_counts.index, distance_counts.values, color=colors, alpha=0.8)

ax.set_title('Trip Distance Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Distance Category', fontsize=12)
ax.set_ylabel('Number of trips', fontsize=12)
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

plt.savefig('simple_distance_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n=== SIMPLE VISUALIZATIONS COMPLETED ===")
print("Generated files:")
print("1. simple_geographic_coverage.png")
print("2. simple_daily_demand.png") 
print("3. simple_hourly_demand.png")
print("4. simple_distance_distribution.png")
print("\nAll visualizations created successfully!")
