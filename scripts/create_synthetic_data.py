#!/usr/bin/env python3
"""
Create synthetic timeseries data for testing the basin similarity pipeline.
This generates realistic streamflow data based on basin attributes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_streamflow(basin_attrs, n_years=5):
    """Generate synthetic streamflow data based on basin attributes."""
    
    # Base parameters from basin attributes
    precip_mean = basin_attrs.get('precip_mean', 800)  # mm/year
    temp_mean = basin_attrs.get('temp_mean', 10)       # °C
    aridity = basin_attrs.get('aridity_index', 1.0)    # aridity index
    forest_frac = basin_attrs.get('forest_frac', 0.3)   # forest fraction
    
    # Generate daily dates
    start_date = datetime(2015, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_years * 365)]
    
    # Seasonal patterns
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Base seasonal streamflow (higher in spring/summer for temperate)
    seasonal_base = 1.0 + 0.5 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
    
    # Temperature effect (more evaporation in summer)
    temp_effect = 1.0 - 0.1 * (temp_mean - 10) / 20
    
    # Precipitation effect
    precip_effect = precip_mean / 800  # normalize to ~800mm
    
    # Aridity effect (more arid = lower base flow)
    aridity_effect = 1.0 / (1.0 + aridity)
    
    # Forest effect (forests reduce peak flows)
    forest_effect = 1.0 - 0.2 * forest_frac
    
    # Combine effects
    base_flow = seasonal_base * temp_effect * precip_effect * aridity_effect * forest_effect
    
    # Add random noise and autocorrelation
    noise = np.random.normal(0, 0.1, len(dates))
    for i in range(1, len(noise)):
        noise[i] = 0.7 * noise[i-1] + 0.3 * noise[i]
    
    # Generate streamflow (m³/s)
    streamflow = base_flow * (1 + noise) * 10  # Scale to reasonable values
    
    # Ensure positive values
    streamflow = np.maximum(streamflow, 0.1)
    
    return dates, streamflow

def create_synthetic_timeseries():
    """Create synthetic timeseries.csv for all basins."""
    
    # Load existing attributes
    attrs = pd.read_csv("data/attributes.csv")
    
    print(f"Creating synthetic timeseries for {len(attrs)} basins...")
    
    all_data = []
    
    for _, basin in attrs.iterrows():
        gauge_id = basin['gauge_id']
        
        # Generate synthetic data
        dates, streamflow = generate_synthetic_streamflow(basin.to_dict())
        
        # Create DataFrame for this basin
        basin_data = pd.DataFrame({
            'gauge_id': gauge_id,
            'date': dates,
            'streamflow': streamflow,
            'precipitation': np.random.exponential(2.0, len(dates)),  # mm/day
            'temperature': basin.get('temp_mean', 10) + np.random.normal(0, 5, len(dates))  # °C
        })
        
        all_data.append(basin_data)
    
    # Combine all basins
    timeseries = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    timeseries.to_csv("data/timeseries.csv", index=False)
    
    print(f"[OK] Created synthetic timeseries with {len(timeseries)} records")
    print(f"   Date range: {timeseries['date'].min()} to {timeseries['date'].max()}")
    print(f"   Basins: {timeseries['gauge_id'].nunique()}")
    print(f"   Columns: {list(timeseries.columns)}")
    
    return timeseries

if __name__ == "__main__":
    create_synthetic_timeseries()
