#!/usr/bin/env python3
"""
Interactive Basin Similarity and Drought Prediction
Allows you to input basin characteristics and get drought predictions.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_basin_data():
    """Load existing basin data for comparison."""
    try:
        attrs = pd.read_csv("data/attributes.csv")
        ts = pd.read_csv("data/timeseries.csv")
        
        # Process timeseries to get drought statistics
        ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
        ts = ts.dropna(subset=["date","streamflow"]).sort_values(["gauge_id","date"])
        
        # Monthly aggregation and drought labeling
        ts["month"] = ts["date"].dt.to_period("M")
        monthly = ts.groupby(["gauge_id","month"], as_index=False)["streamflow"].mean()
        
        # SRI-like standardization per basin
        def sri(df):
            mu, sd = df["streamflow"].mean(), df["streamflow"].std()
            df["SRI"] = (df["streamflow"] - mu) / (sd if sd and sd>0 else 1.0)
            return df
        
        monthly = monthly.groupby("gauge_id", group_keys=False).apply(sri)
        monthly["drought_flag"] = (monthly["SRI"] < -1.0).astype(int)
        
        # Calculate basin statistics
        basin_stats = monthly.groupby('gauge_id').agg({
            'SRI': ['mean', 'std'],
            'streamflow': ['mean', 'std'],
            'drought_flag': ['sum', 'mean']
        }).round(3)
        
        # Flatten column names
        basin_stats.columns = ['_'.join(col).strip() for col in basin_stats.columns]
        basin_stats = basin_stats.reset_index()
        
        return attrs, basin_stats, monthly
        
    except Exception as e:
        print(f"Warning: Could not load existing data: {e}")
        return None, None, None

def calculate_drought_probability(basin_attrs, reference_data=None):
    """Calculate drought probability based on basin characteristics."""
    
    # Default drought probability
    base_prob = 0.2  # 20% baseline
    
    # Adjust based on characteristics
    adjustments = []
    
    # Aridity index effect (higher aridity = more drought)
    if 'aridity_index' in basin_attrs:
        aridity = basin_attrs['aridity_index']
        if aridity > 1.5:  # Very arid
            adjustments.append(0.3)
        elif aridity > 1.0:  # Arid
            adjustments.append(0.15)
        elif aridity < 0.5:  # Humid
            adjustments.append(-0.1)
    
    # Precipitation effect
    if 'precip_mean' in basin_attrs:
        precip = basin_attrs['precip_mean']
        if precip < 400:  # Very dry
            adjustments.append(0.25)
        elif precip < 600:  # Dry
            adjustments.append(0.1)
        elif precip > 1200:  # Very wet
            adjustments.append(-0.15)
    
    # Temperature effect
    if 'temp_mean' in basin_attrs:
        temp = basin_attrs['temp_mean']
        if temp > 20:  # Hot climate
            adjustments.append(0.1)
        elif temp < 5:  # Cold climate
            adjustments.append(-0.05)
    
    # Forest fraction effect
    if 'forest_frac' in basin_attrs:
        forest = basin_attrs['forest_frac']
        if forest > 0.7:  # Heavily forested
            adjustments.append(-0.1)
        elif forest < 0.1:  # No forest
            adjustments.append(0.1)
    
    # Elevation effect
    if 'elev_mean' in basin_attrs:
        elev = basin_attrs['elev_mean']
        if elev > 2000:  # High elevation
            adjustments.append(0.05)
        elif elev < 100:  # Low elevation
            adjustments.append(0.05)
    
    # Calculate final probability
    total_adjustment = sum(adjustments)
    drought_prob = base_prob + total_adjustment
    drought_prob = max(0.05, min(0.8, drought_prob))  # Clamp between 5% and 80%
    
    return drought_prob, adjustments

def predict_streamflow(basin_attrs):
    """Predict average streamflow based on basin characteristics."""
    
    # Base streamflow (m³/s)
    base_flow = 5.0
    
    # Adjustments based on characteristics
    flow_multiplier = 1.0
    
    # Precipitation effect
    if 'precip_mean' in basin_attrs:
        precip = basin_attrs['precip_mean']
        flow_multiplier *= (precip / 800) ** 0.8  # Non-linear relationship
    
    # Aridity effect
    if 'aridity_index' in basin_attrs:
        aridity = basin_attrs['aridity_index']
        flow_multiplier *= 1.0 / (1.0 + aridity * 0.3)
    
    # Forest effect (forests reduce peak flows but maintain base flow)
    if 'forest_frac' in basin_attrs:
        forest = basin_attrs['forest_frac']
        flow_multiplier *= 0.8 + 0.4 * forest  # Range: 0.8 to 1.2
    
    # Temperature effect (evaporation)
    if 'temp_mean' in basin_attrs:
        temp = basin_attrs['temp_mean']
        flow_multiplier *= 1.0 - (temp - 10) * 0.02  # 2% reduction per °C above 10°C
    
    predicted_flow = base_flow * flow_multiplier
    return max(0.1, predicted_flow)  # Minimum flow

def find_similar_basins(basin_attrs, reference_data, n_similar=3):
    """Find similar basins based on characteristics."""
    
    if reference_data is None:
        return []
    
    # Calculate similarity scores
    similarities = []
    
    for _, basin in reference_data.iterrows():
        score = 0
        n_features = 0
        
        # Compare each characteristic
        for attr, value in basin_attrs.items():
            if attr in basin.index and not pd.isna(basin[attr]):
                # Normalized difference
                diff = abs(value - basin[attr])
                if attr in ['precip_mean', 'temp_mean', 'elev_mean']:
                    # Scale by typical ranges
                    if attr == 'precip_mean':
                        diff = diff / 500  # Scale by 500mm
                    elif attr == 'temp_mean':
                        diff = diff / 10   # Scale by 10°C
                    elif attr == 'elev_mean':
                        diff = diff / 1000  # Scale by 1000m
                elif attr == 'aridity_index':
                    diff = diff / 2  # Scale by 2
                elif attr == 'forest_frac':
                    diff = diff / 1  # Scale by 1
                
                score += diff
                n_features += 1
        
        if n_features > 0:
            avg_score = score / n_features
            similarities.append((basin['gauge_id'], avg_score, basin))
    
    # Sort by similarity (lower score = more similar)
    similarities.sort(key=lambda x: x[1])
    
    return similarities[:n_similar]

def interactive_prediction():
    """Interactive basin prediction interface."""
    
    print("=" * 60)
    print("INTERACTIVE BASIN DROUGHT PREDICTION")
    print("=" * 60)
    print("Enter basin characteristics to predict drought probability and streamflow.")
    print("Press Enter to use default values or 'quit' to exit.\n")
    
    # Load reference data
    attrs, basin_stats, monthly_data = load_basin_data()
    
    if attrs is not None:
        print(f"Loaded reference data: {len(attrs)} basins")
        print("Available characteristics in reference data:")
        print(f"  {list(attrs.columns)}")
        print()
    
    while True:
        print("-" * 40)
        print("BASIN CHARACTERISTICS INPUT")
        print("-" * 40)
        
        # Get basin characteristics
        basin_attrs = {}
        
        # Common characteristics with descriptions
        characteristics = {
            'aridity_index': ('Aridity Index (0-2, higher = more arid)', 1.0),
            'precip_mean': ('Mean Annual Precipitation (mm/year)', 800),
            'temp_mean': ('Mean Annual Temperature (°C)', 10),
            'elev_mean': ('Mean Elevation (meters)', 500),
            'forest_frac': ('Forest Fraction (0-1)', 0.3),
            'slope_mean': ('Mean Slope (degrees)', 10)
        }
        
        for attr, (description, default) in characteristics.items():
            while True:
                try:
                    value = input(f"{attr} ({description}) [default: {default}]: ").strip()
                    if value.lower() == 'quit':
                        print("Goodbye!")
                        return
                    elif value == '':
                        basin_attrs[attr] = default
                        break
                    else:
                        basin_attrs[attr] = float(value)
                        break
                except ValueError:
                    print("Please enter a valid number or press Enter for default.")
        
        # Optional additional characteristics
        print("\nOptional additional characteristics (press Enter to skip):")
        additional = {
            'region': ('Geographic Region', 'temperate'),
            'climate_class': ('Climate Classification', 'Cfb')
        }
        
        for attr, (description, default) in additional.items():
            value = input(f"{attr} ({description}) [default: {default}]: ").strip()
            if value.lower() == 'quit':
                print("Goodbye!")
                return
            elif value == '':
                basin_attrs[attr] = default
            else:
                basin_attrs[attr] = value
        
        # Make predictions
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        
        # Drought probability
        drought_prob, adjustments = calculate_drought_probability(basin_attrs, basin_stats)
        print(f"Drought Probability: {drought_prob:.1%}")
        
        if adjustments:
            print("Adjustments made:")
            for i, adj in enumerate(adjustments):
                if adj > 0:
                    print(f"  +{adj:.1%} (increased drought risk)")
                else:
                    print(f"  {adj:.1%} (decreased drought risk)")
        
        # Streamflow prediction
        predicted_flow = predict_streamflow(basin_attrs)
        print(f"Predicted Average Streamflow: {predicted_flow:.2f} m³/s")
        
        # Drought classification
        if drought_prob > 0.4:
            classification = "HIGH DROUGHT RISK"
        elif drought_prob > 0.25:
            classification = "MODERATE DROUGHT RISK"
        else:
            classification = "LOW DROUGHT RISK"
        
        print(f"Drought Classification: {classification}")
        
        # Find similar basins
        if basin_stats is not None:
            similar_basins = find_similar_basins(basin_attrs, basin_stats)
            if similar_basins:
                print(f"\nMost Similar Basins:")
                for i, (basin_id, score, basin_data) in enumerate(similar_basins, 1):
                    print(f"  {i}. {basin_id} (similarity score: {score:.3f})")
                    if 'drought_flag_mean' in basin_data:
                        print(f"     Historical drought rate: {basin_data['drought_flag_mean']:.1%}")
        
        # Save results
        save_results = input("\nSave these results to file? (y/n): ").strip().lower()
        if save_results == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'timestamp': timestamp,
                'basin_characteristics': basin_attrs,
                'drought_probability': drought_prob,
                'predicted_streamflow': predicted_flow,
                'drought_classification': classification,
                'adjustments': adjustments
            }
            
            filename = f"basin_prediction_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {filename}")
        
        # Continue or quit
        continue_pred = input("\nMake another prediction? (y/n): ").strip().lower()
        if continue_pred != 'y':
            print("Goodbye!")
            break

if __name__ == "__main__":
    interactive_prediction()
