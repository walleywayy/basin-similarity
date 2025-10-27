#!/usr/bin/env python3
"""
Predict Drought and Streamflow for Unseen Basins
Loads trained models and makes predictions on new basin characteristics.
"""

import json
import pandas as pd
import numpy as np
import joblib
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_models(models_dir="models"):
    """Load all trained models and preprocessing objects."""
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' not found. Run predict_droughts.py first.")
    
    print(f"Loading models from {models_dir}/...")
    
    # Load models
    clf = joblib.load(f"{models_dir}/drought_classifier.joblib")
    reg = joblib.load(f"{models_dir}/streamflow_regressor.joblib")
    scaler = joblib.load(f"{models_dir}/feature_scaler.joblib")
    pca = joblib.load(f"{models_dir}/pca_transformer.joblib")
    metadata = joblib.load(f"{models_dir}/model_metadata.joblib")
    
    print("✅ All models loaded successfully")
    return clf, reg, scaler, pca, metadata

def create_basin_features(basin_attrs, metadata):
    """Create feature vector from basin attributes."""
    feature_names = metadata['feature_names']
    
    # Create a DataFrame with the expected features
    features_df = pd.DataFrame(index=[0])
    
    # Fill in provided attributes
    for feature in feature_names:
        if feature in basin_attrs:
            features_df[feature] = basin_attrs[feature]
        else:
            # Use median value from training data as default
            features_df[feature] = 0.0  # Will be filled with median during preprocessing
    
    return features_df

def predict_basin(basin_attrs, models_dir="models", verbose=True):
    """
    Predict drought probability and streamflow for a new basin.
    
    Parameters:
    -----------
    basin_attrs : dict
        Dictionary containing basin attributes (e.g., {'aridity_index': 0.5, 'slope_mean': 15.2, ...})
    models_dir : str
        Path to directory containing trained models
    verbose : bool
        Whether to print detailed output
        
    Returns:
    --------
    dict : Prediction results containing drought_prob, streamflow_pred, and basin_similarity
    """
    
    # Load models
    clf, reg, scaler, pca, metadata = load_models(models_dir)
    
    if verbose:
        print(f"\nMaking predictions for basin with attributes:")
        for key, value in basin_attrs.items():
            print(f"  {key}: {value}")
    
    # Create feature vector
    features_df = create_basin_features(basin_attrs, metadata)
    
    # Handle missing values with median imputation
    # Note: In a real scenario, you'd want to use the same imputation strategy as training
    features_df = features_df.fillna(0.0)
    
    # Scale features
    X_scaled = scaler.transform(features_df)
    
    # Make predictions
    drought_prob = clf.predict_proba(X_scaled)[0, 1]  # Probability of drought
    drought_class = clf.predict(X_scaled)[0]  # Binary classification
    streamflow_pred = reg.predict(X_scaled)[0]  # Predicted streamflow
    
    # Compute basin similarity (PCA embedding)
    pca_embedding = pca.transform(X_scaled)
    
    # Results
    results = {
        'drought_probability': float(drought_prob),
        'drought_class': int(drought_class),
        'predicted_streamflow': float(streamflow_pred),
        'basin_similarity_pc1': float(pca_embedding[0, 0]),
        'basin_similarity_pc2': float(pca_embedding[0, 1]),
        'model_metadata': {
            'training_drought_rate': float(metadata['drought_rate']),
            'training_n_basins': int(metadata['n_basins']),
            'training_n_samples': int(metadata['n_samples'])
        }
    }
    
    if verbose:
        print(f"\nPREDICTION RESULTS:")
        print(f"  Drought Probability: {drought_prob:.3f} ({drought_prob*100:.1f}%)")
        print(f"  Drought Classification: {'DROUGHT' if drought_class else 'NORMAL'}")
        print(f"  Predicted Streamflow: {streamflow_pred:.3f}")
        print(f"  Basin Similarity (PC1): {pca_embedding[0, 0]:.3f}")
        print(f"  Basin Similarity (PC2): {pca_embedding[0, 1]:.3f}")
        print(f"\nModel trained on {metadata['n_basins']} basins with {metadata['drought_rate']:.1%} drought rate")
    
    return results

def predict_from_json(json_file, models_dir="models"):
    """Predict from a JSON file containing basin attributes."""
    with open(json_file, 'r') as f:
        basin_attrs = json.load(f)
    
    return predict_basin(basin_attrs, models_dir)

def create_example_basin():
    """Create an example basin for demonstration."""
    return {
        'aridity_index': 0.8,  # Semi-arid
        'slope_mean': 12.5,    # Moderate slope
        'elev_mean': 850.0,    # Mid-elevation
        'forest_frac': 0.3,    # 30% forest cover
        'temp_mean': 8.5,      # Cool temperate
        'precip_mean': 650.0,  # Moderate precipitation
        'region': 'temperate',
        'climate_class': 'Cfb'
    }

def main():
    parser = argparse.ArgumentParser(description="Predict drought and streamflow for unseen basins")
    parser.add_argument('--json-file', type=str, help='JSON file with basin attributes')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory containing trained models')
    parser.add_argument('--example', action='store_true', help='Run prediction on example basin')
    parser.add_argument('--save-example', action='store_true', help='Save example basin JSON file')
    
    args = parser.parse_args()
    
    if args.save_example:
        example_basin = create_example_basin()
        with open('example_basin.json', 'w') as f:
            json.dump(example_basin, f, indent=2)
        print("✅ Example basin saved to example_basin.json")
        return
    
    if args.example:
        print("Running prediction on example basin...")
        example_basin = create_example_basin()
        results = predict_basin(example_basin, args.models_dir)
        
        print(f"\nExample basin attributes:")
        for key, value in example_basin.items():
            print(f"  {key}: {value}")
        
        return results
    
    if args.json_file:
        if not os.path.exists(args.json_file):
            print(f"Error: JSON file '{args.json_file}' not found")
            return
        
        print(f"Loading basin attributes from {args.json_file}...")
        results = predict_from_json(args.json_file, args.models_dir)
        return results
    
    # Interactive mode
    print("UNSEEN BASIN PREDICTION")
    print("=" * 50)
    print("Enter basin attributes (press Enter to use default values):")
    
    # Get user input for basin attributes
    basin_attrs = {}
    
    # Common attributes with defaults
    attributes = {
        'aridity_index': 'Aridity index (0-2, higher = more arid)',
        'slope_mean': 'Mean slope (degrees)',
        'elev_mean': 'Mean elevation (meters)',
        'forest_frac': 'Forest fraction (0-1)',
        'temp_mean': 'Mean temperature (°C)',
        'precip_mean': 'Mean precipitation (mm/year)',
        'region': 'Geographic region',
        'climate_class': 'Climate classification'
    }
    
    for attr, description in attributes.items():
        value = input(f"{attr} ({description}): ").strip()
        if value:
            try:
                # Try to convert to float, fall back to string
                basin_attrs[attr] = float(value) if '.' in value or value.isdigit() else value
            except ValueError:
                basin_attrs[attr] = value
    
    if not basin_attrs:
        print("No attributes provided. Using example basin...")
        basin_attrs = create_example_basin()
    
    # Make prediction
    results = predict_basin(basin_attrs, args.models_dir)
    
    print(f"\n🎉 Prediction complete!")
    print(f"Use --save-example to create a template JSON file for batch predictions.")

if __name__ == "__main__":
    main()
