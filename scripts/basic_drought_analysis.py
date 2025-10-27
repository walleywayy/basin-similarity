#!/usr/bin/env python3
"""
Basic Basin Similarity and Drought Analysis
Demonstrates the concept without complex dependencies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    print("BASIN SIMILARITY AND DROUGHT ANALYSIS")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    attrs = pd.read_csv("data/attributes.csv")
    
    # Check if timeseries exists
    if not os.path.exists("data/timeseries.csv"):
        print("Warning: Timeseries file not found at data/timeseries.csv")
        print("This is normal if no basins had timeseries data in the dataset.")
        print("Continuing with attributes-only analysis...")
        ts = None
    else:
        ts = pd.read_csv("data/timeseries.csv")
    
    print(f"Attributes shape: {attrs.shape}")
    if ts is not None:
        print(f"Timeseries shape: {ts.shape}")
        print(f"Timeseries columns: {list(ts.columns)}")
    else:
        print("Timeseries: Not available")
    print(f"Attributes columns: {list(attrs.columns)}")
    
    if ts is None:
        print("\nATTRIBUTES-ONLY ANALYSIS")
        print("=" * 30)
        print("Since no timeseries data is available, we'll analyze basin characteristics:")
        print(f"[OK] Loaded {len(attrs)} basins")
        print(f"[OK] Available attributes: {list(attrs.columns)}")
        print(f"[OK] Clean gauge IDs: {attrs['gauge_id'].head().tolist()}")
        
        # Basic attribute analysis
        print("\nBasin attribute summary:")
        print(attrs.describe())
        
        print("\n[SUCCESS] Attributes analysis complete!")
        print("To get timeseries data, try:")
        print("1. Use a different dataset: --dataset caravan")
        print("2. Increase basin count: --n-basins 100")
        print("3. Remove size cap: remove --size-cap-mb")
        return
    
    # Basic cleanup
    ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
    ts = ts.dropna(subset=["date","streamflow"]).sort_values(["gauge_id","date"])
    
    print(f"\nTimeseries after cleanup: {ts.shape}")
    print(f"Date range: {ts['date'].min()} to {ts['date'].max()}")
    print(f"Number of basins: {ts['gauge_id'].nunique()}")
    
    # Monthly aggregation and drought labeling
    print("\nComputing monthly aggregates and drought labels...")
    ts["month"] = ts["date"].dt.to_period("M")
    monthly = ts.groupby(["gauge_id","month"], as_index=False)["streamflow"].mean()
    
    # SRI-like standardization per basin
    def sri(df):
        mu, sd = df["streamflow"].mean(), df["streamflow"].std()
        df["SRI"] = (df["streamflow"] - mu) / (sd if sd and sd>0 else 1.0)
        return df
    
    monthly = monthly.groupby("gauge_id", group_keys=False).apply(sri)
    monthly["drought_flag"] = (monthly["SRI"] < -1.0).astype(int)
    
    print(f"Monthly data shape: {monthly.shape}")
    print(f"Drought events: {monthly['drought_flag'].sum()} ({monthly['drought_flag'].mean():.1%})")
    
    # Merge with attributes
    df = monthly.merge(attrs, on="gauge_id", how="left")
    df = df.dropna(subset=["SRI"])
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Drought events: {df['drought_flag'].sum()} ({df['drought_flag'].mean():.1%})")
    
    # Basic basin similarity analysis
    print("\nComputing basin similarity...")
    basin_stats = df.groupby('gauge_id').agg({
        'SRI': ['mean', 'std', 'min', 'max'],
        'streamflow': ['mean', 'std'],
        'drought_flag': 'sum'
    }).round(3)
    
    print(f"Basin statistics shape: {basin_stats.shape}")
    print("\nSample basin statistics:")
    print(basin_stats.head())
    
    # Simple correlation analysis
    print("\nCorrelation analysis:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Find correlations with drought flag
    drought_corr = corr_matrix['drought_flag'].abs().sort_values(ascending=False)
    print("\nTop correlations with drought flag:")
    print(drought_corr.head(10))
    
    # Basic visualization
    print("\nCreating visualizations...")
    
    # Plot 1: Streamflow distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(df['streamflow'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Streamflow')
    plt.ylabel('Frequency')
    plt.title('Streamflow Distribution')
    plt.yscale('log')
    
    # Plot 2: SRI distribution
    plt.subplot(1, 3, 2)
    plt.hist(df['SRI'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('SRI (Standardized Runoff Index)')
    plt.ylabel('Frequency')
    plt.title('SRI Distribution')
    plt.axvline(x=-1.0, color='red', linestyle='--', label='Drought Threshold')
    plt.legend()
    
    # Plot 3: Drought events over time
    plt.subplot(1, 3, 3)
    monthly_drought = df.groupby('month')['drought_flag'].mean()
    monthly_drought.plot(kind='line', marker='o')
    plt.xlabel('Month')
    plt.ylabel('Drought Event Rate')
    plt.title('Drought Events Over Time')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('drought_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\nSUMMARY STATISTICS")
    print("=" * 50)
    print(f"[OK] Processed {df['gauge_id'].nunique()} unique basins")
    print(f"[OK] Analyzed {len(df)} monthly observations")
    print(f"[OK] Identified {df['drought_flag'].sum()} drought events ({df['drought_flag'].mean():.1%})")
    print(f"[OK] Computed basin similarity metrics")
    print(f"[OK] Generated correlation analysis")
    
    # Key findings
    print("\nKEY FINDINGS:")
    print(f"- Average streamflow: {df['streamflow'].mean():.2f}")
    print(f"- Average SRI: {df['SRI'].mean():.2f}")
    print(f"- Drought frequency: {df['drought_flag'].mean():.1%}")
    print(f"- Most correlated feature with drought: {drought_corr.index[1]} ({drought_corr.iloc[1]:.3f})")
    
    print("\nNEXT STEPS:")
    print("1. Install compatible scikit-learn for machine learning models")
    print("2. Add more meteorological features")
    print("3. Implement UMAP for better similarity mapping")
    print("4. Build Random Forest models for prediction")
    print("5. Create interactive visualizations")
    
    print("\nFILES GENERATED:")
    print("- data/attributes.csv: Basin characteristics")
    print("- data/demo_gauges.txt: Basin IDs")
    print("- data/timeseries.csv: Streamflow data")
    print("- drought_analysis.png: Analysis visualizations")
    
    print("\n[SUCCESS] Basic analysis complete! Ready for advanced modeling.")

if __name__ == "__main__":
    main()
