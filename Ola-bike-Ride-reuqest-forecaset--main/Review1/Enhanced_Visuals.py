import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth in kilometers"""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

print("=== CREATING ENHANCED VISUALIZATIONS ===")

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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

# 1. ENHANCED Geographic Coverage with Density Analysis
print("Creating enhanced geographic coverage map...")
fig, ax = plt.subplots(figsize=(14, 10))

# Create density-based coloring
from scipy.stats import gaussian_kde

# Sample data for performance (take every 100th point)
sample_df = df.iloc[::100].copy()

# Calculate density
coords = np.vstack([sample_df['pick_lng'], sample_df['pick_lat']])
density = gaussian_kde(coords)(coords)

# Create scatter plot with density-based colors
scatter = ax.scatter(sample_df['pick_lng'], sample_df['pick_lat'], 
                    c=density, cmap='RdYlGn_r', s=1, alpha=0.6)

# Add Bangalore center marker
bangalore_lat, bangalore_lng = 12.9716, 77.5946
ax.scatter(bangalore_lng, bangalore_lat, marker='*', s=500, c='red', 
          edgecolors='black', linewidth=2, label='Bangalore Center')

# Add density legend
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Service Density', fontsize=12)

ax.set_title('Enhanced Geographic Service Coverage', fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('enhanced_geographic_coverage.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. ENHANCED Daily Demand Distribution
print("Creating enhanced daily demand distribution...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Pie chart
day_counts = df['day_of_week'].value_counts()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = day_counts.reindex(days_order)

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
wedges, texts, autotexts = ax1.pie(day_counts.values, labels=day_counts.index, 
                                  autopct='%1.1f%%', startangle=90, colors=colors,
                                  explode=[0.05 if day == 'Friday' else 0 for day in days_order])

# Highlight Friday
for autotext in autotexts:
    if autotext.get_text() == '16.0%':  # Friday percentage
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

ax1.set_title('Daily Demand Distribution', fontsize=14, fontweight='bold')

# Bar chart for comparison
bars = ax2.bar(range(len(day_counts)), day_counts.values, color=colors)
ax2.set_title('Daily Demand Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Day of Week', fontsize=12)
ax2.set_ylabel('Number of Trips', fontsize=12)
ax2.set_xticks(range(len(day_counts)))
ax2.set_xticklabels(day_counts.index, rotation=45)
ax2.grid(True, alpha=0.3)

# Highlight Friday bar
bars[4].set_edgecolor('black')
bars[4].set_linewidth(3)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('enhanced_daily_demand.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. ENHANCED Hourly Demand Pattern
print("Creating enhanced hourly demand pattern...")
fig, ax = plt.subplots(figsize=(16, 8))

hour_counts = df['hour'].value_counts().sort_index()

# Define peak hours (0-2 AM)
peak_hours = [0, 1, 2]
colors = ['red' if hour in peak_hours else '#4ECDC4' for hour in hour_counts.index]

bars = ax.bar(hour_counts.index, hour_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

# Add peak hours annotation
ax.annotate('Peak Hours\n(0-2 AM)', xy=(1, hour_counts[1]), xytext=(3, hour_counts[1] + 20),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, fontweight='bold', color='red')

# Add secondary peak annotation
morning_peak = hour_counts[9]  # 9 AM
ax.annotate('Morning Peak', xy=(9, morning_peak), xytext=(11, morning_peak + 15),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=10, color='blue')

ax.set_title('Enhanced Hourly Demand Pattern', fontsize=16, fontweight='bold')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Number of Trips', fontsize=12)
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', alpha=0.8, label='Peak Hours (0-2 AM)'),
                   Patch(facecolor='#4ECDC4', alpha=0.8, label='Regular Hours')]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('enhanced_hourly_demand.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. ENHANCED Trip Distance Distribution
print("Creating enhanced trip distance distribution...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Categorize distances
df['distance_category'] = pd.cut(df['trip_distance'], 
                                bins=[0, 5, 10, 15, float('inf')], 
                                labels=['0-5km', '5-10km', '10-15km', '15+km'])

distance_counts = df['distance_category'].value_counts().sort_index()

# Bar chart
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax1.bar(distance_counts.index, distance_counts.values, color=colors, alpha=0.8, edgecolor='black')

ax1.set_title('Trip Distance Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Distance Category', fontsize=12)
ax1.set_ylabel('Number of Trips', fontsize=12)
ax1.grid(True, alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

# Histogram
ax2.hist(df['trip_distance'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax2.axvline(df['trip_distance'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {df["trip_distance"].mean():.2f} km')
ax2.axvline(df['trip_distance'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median: {df["trip_distance"].median():.2f} km')

ax2.set_title('Trip Distance Histogram', fontsize=14, fontweight='bold')
ax2.set_xlabel('Distance (km)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('enhanced_distance_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. BONUS: Heatmap of Hour vs Day of Week
print("Creating hourly vs daily heatmap...")
fig, ax = plt.subplots(figsize=(14, 8))

# Create pivot table for heatmap
heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex(days_order)

# Create heatmap
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
            cbar_kws={'label': 'Number of Trips'}, ax=ax)

ax.set_title('Demand Heatmap: Hour vs Day of Week', fontsize=16, fontweight='bold')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Day of Week', fontsize=12)

plt.tight_layout()
plt.savefig('demand_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n=== ALL ENHANCED VISUALIZATIONS COMPLETED ===")
print("Generated files:")
print("1. enhanced_geographic_coverage.png - Density-based geographic map")
print("2. enhanced_daily_demand.png - Daily demand pie chart + bar chart")
print("3. enhanced_hourly_demand.png - Hourly pattern with peak annotations")
print("4. enhanced_distance_distribution.png - Distance categories + histogram")
print("5. demand_heatmap.png - Hour vs day heatmap")
print("\nAll visualizations include enhanced styling, annotations, and insights!")
