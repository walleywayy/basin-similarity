#!/usr/bin/env python3
"""
Basin Similarity and Drought Prediction Script
Demonstrates the complete pipeline from data loading to model training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    print("BASIN SIMILARITY AND DROUGHT PREDICTION")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    attrs = pd.read_csv("data/attributes.csv")
    ts = pd.read_csv("data/timeseries.csv")
    
    print(f"Attributes shape: {attrs.shape}")
    print(f"Timeseries shape: {ts.shape}")
    print(f"Attributes columns: {list(attrs.columns)}")
    print(f"Timeseries columns: {list(ts.columns)}")
    
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
    
    # Extract meteorological features if available
    met_cols = [c for c in ts.columns if c.lower() in ['precipitation', 'temperature']]
    print(f"Available met columns: {met_cols}")
    
    if met_cols:
        met = ts.groupby(["gauge_id","month"], as_index=False)[met_cols].mean()
        met = met.rename(columns={"precipitation":"prcp", "temperature":"tavg"})
        
        # Lagged features (3-month rolling mean)
        if "prcp" in met.columns:
            met["prcp_3mo"] = met.groupby("gauge_id")["prcp"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        if "tavg" in met.columns:
            met["tavg_3mo"] = met.groupby("gauge_id")["tavg"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        
        print(f"Meteorological data shape: {met.shape}")
    else:
        print("No meteorological data available - using attributes only")
        met = pd.DataFrame()
    
    # Merge attributes + met + target
    if not met.empty:
        df = monthly.merge(met, on=["gauge_id","month"], how="left")
    else:
        df = monthly.copy()
        
    df = df.merge(attrs, on="gauge_id", how="left")
    df = df.dropna(subset=["SRI"])  # target available
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Drought events: {df['drought_flag'].sum()} ({df['drought_flag'].mean():.1%})")
    
    # Basin similarity mapping
    print("\nComputing basin similarity...")
    feature_cols = [c for c in df.columns if c not in ['SRI','drought_flag','streamflow','month','gauge_id','source','orig_id']]
    print(f"Using features for similarity: {feature_cols}")
    
    # Get unique basins with their attributes
    basin_features = df.groupby('gauge_id')[feature_cols].first().reset_index()
    print(f"Basin features shape: {basin_features.shape}")
    
    # Handle missing values
    basin_features = basin_features.fillna(basin_features.median())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(basin_features[feature_cols])
    
    # PCA for similarity mapping
    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_embedding = pca.fit_transform(X_scaled)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Plot similarity map
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_embedding[:, 0], pca_embedding[:, 1], c=basin_features.index, cmap='viridis', alpha=0.7)
    plt.title('Basin Similarity Map (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(scatter, label='Basin Index')
    plt.tight_layout()
    plt.savefig('basin_similarity_map.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Basin similarity analysis complete!")
    print(f"Found {len(basin_features)} unique basins")
    
    # Drought prediction model
    print("\nTraining drought prediction model...")
    X = df.drop(columns=["SRI","drought_flag","streamflow","month","gauge_id","source","orig_id"])
    y = df["drought_flag"]
    
    # Handle missing values
    X = X.fillna(X.median())
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train drought rate: {y_train.mean():.1%}")
    print(f"Test drought rate: {y_test.mean():.1%}")
    
    # Train Random Forest for drought prediction
    clf = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Results
    print("\nDROUGHT PREDICTION RESULTS")
    print("=" * 50)
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Streamflow prediction model
    print("\nTraining streamflow prediction model...")
    X_reg = df.drop(columns=["SRI","drought_flag","streamflow","month","gauge_id","source","orig_id"])
    y_reg = df["streamflow"]
    
    # Handle missing values
    X_reg = X_reg.fillna(X_reg.median())
    
    # Train/test split
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, random_state=42, test_size=0.2)
    
    # Train model
    reg = RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=42)
    reg.fit(X_train_reg, y_train_reg)
    
    # Predictions
    y_pred_reg = reg.predict(X_test_reg)
    
    # Results
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    print("\nSTREAMFLOW PREDICTION RESULTS")
    print("=" * 50)
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {np.sqrt(mse):.3f}")
    print(f"Mean Actual: {y_test_reg.mean():.3f}")
    print(f"Mean Predicted: {y_pred_reg.mean():.3f}")
    
    # Summary
    print("\nSUMMARY")
    print("=" * 50)
    print(f"✅ Processed {len(basin_features)} unique basins")
    print(f"✅ Generated similarity map using PCA")
    print(f"✅ Trained drought prediction model (Random Forest)")
    print(f"✅ Trained streamflow prediction model (Random Forest)")
    print(f"✅ Identified most important features for predictions")
    
    print("\nNEXT STEPS:")
    print("1. Collect more meteorological data for better predictions")
    print("2. Experiment with different similarity metrics")
    print("3. Add temporal features (seasonality, trends)")
    print("4. Validate on out-of-sample basins")
    print("5. Build interactive similarity map visualization")
    
    # Save trained models and preprocessing objects
    print("\nSaving trained models and preprocessing objects...")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save models
    joblib.dump(clf, f"{models_dir}/drought_classifier.joblib")
    joblib.dump(reg, f"{models_dir}/streamflow_regressor.joblib")
    joblib.dump(scaler, f"{models_dir}/feature_scaler.joblib")
    joblib.dump(pca, f"{models_dir}/pca_transformer.joblib")
    
    # Save feature names and metadata
    model_metadata = {
        'feature_names': list(X.columns),
        'target_mean': y_reg.mean(),
        'target_std': y_reg.std(),
        'drought_rate': y.mean(),
        'n_basins': len(basin_features),
        'n_samples': len(df)
    }
    joblib.dump(model_metadata, f"{models_dir}/model_metadata.joblib")
    
    print(f"✅ Models saved to {models_dir}/")
    print(f"  - drought_classifier.joblib: Random Forest for drought prediction")
    print(f"  - streamflow_regressor.joblib: Random Forest for streamflow prediction")
    print(f"  - feature_scaler.joblib: StandardScaler for feature normalization")
    print(f"  - pca_transformer.joblib: PCA for basin similarity mapping")
    print(f"  - model_metadata.joblib: Feature names and model statistics")
    
    print("\nFILES GENERATED:")
    print("- data/attributes.csv: Basin characteristics")
    print("- data/demo_gauges.txt: Basin IDs for similarity analysis")
    print("- data/timeseries.csv: Streamflow and meteorological data")
    print("- basin_similarity_map.png: PCA similarity visualization")
    print(f"- {models_dir}/: Trained models and preprocessing objects")
    
    print("\n🎉 Analysis complete! The basin similarity framework is ready for further development.")
    print("\nNEXT: Use predict_unseen_basin.py to make predictions on new basins!")

if __name__ == "__main__":
    main()
