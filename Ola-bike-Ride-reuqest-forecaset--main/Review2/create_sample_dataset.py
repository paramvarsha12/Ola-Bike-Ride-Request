"""
Create a sample dataset for testing the Ola bike ride request forecasting system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_dataset(n_samples=10000):
    """Create a realistic sample dataset for Ola bike ride requests"""
    
    print("Creating sample dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate timestamps (last 6 months)
    start_date = datetime.now() - timedelta(days=180)
    timestamps = []
    for i in range(n_samples):
        random_days = random.randint(0, 180)
        random_hours = random.randint(0, 23)
        random_minutes = random.randint(0, 59)
        random_seconds = random.randint(0, 59)
        
        timestamp = start_date + timedelta(
            days=random_days, 
            hours=random_hours, 
            minutes=random_minutes, 
            seconds=random_seconds
        )
        timestamps.append(timestamp)
    
    # Generate user IDs
    user_ids = [f"user_{random.randint(1, 1000)}" for _ in range(n_samples)]
    
    # Generate coordinates around Bangalore
    bangalore_lat, bangalore_lng = 12.9716, 77.5946
    
    # Pickup coordinates (within Bangalore metropolitan area)
    pickup_lats = np.random.normal(bangalore_lat, 0.1, n_samples)
    pickup_lngs = np.random.normal(bangalore_lng, 0.1, n_samples)
    
    # Drop coordinates (within reasonable distance)
    drop_lats = pickup_lats + np.random.normal(0, 0.05, n_samples)
    drop_lngs = pickup_lngs + np.random.normal(0, 0.05, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'ts': timestamps,
        'user_id': user_ids,
        'pick_lat': pickup_lats,
        'pick_lng': pickup_lngs,
        'drop_lat': drop_lats,
        'drop_lng': drop_lngs
    })
    
    # Add some realistic patterns
    # More rides during peak hours (7-9 AM, 5-7 PM)
    peak_hours = [7, 8, 9, 17, 18, 19]
    df['hour'] = df['ts'].dt.hour
    df['is_peak'] = df['hour'].isin(peak_hours)
    
    # More rides on weekdays
    df['is_weekday'] = df['ts'].dt.weekday < 5
    
    # Add some noise to make it more realistic
    df['pick_lat'] += np.random.normal(0, 0.01, n_samples)
    df['pick_lng'] += np.random.normal(0, 0.01, n_samples)
    df['drop_lat'] += np.random.normal(0, 0.01, n_samples)
    df['drop_lng'] += np.random.normal(0, 0.01, n_samples)
    
    print(f"Created dataset with {len(df)} samples")
    print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
    print(f"Unique users: {df['user_id'].nunique()}")
    
    return df

if __name__ == "__main__":
    # Create sample dataset
    df = create_sample_dataset(10000)
    
    # Save to CSV
    df.to_csv('ct_rr.csv', index=False)
    print("Sample dataset saved as 'ct_rr.csv'")
    
    # Display sample
    print("\nSample data:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
