#!/usr/bin/env python3
"""
Fetch a subset of the Caravan dataset from Zenodo.

This script downloads only specific attributes and time series for a random
sample of 100 basins, avoiding downloading the full 12.5GB dataset.
"""

import os
import requests
import zipfile
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict
import tempfile
import io
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Zenodo API URLs
ZENODO_RECORD = "7540792"
ZENODO_BASE_URL = "https://zenodo.org/api/records"
ZENODO_FILES_URL = f"{ZENODO_BASE_URL}/{ZENODO_RECORD}/files"

# Required columns from attributes files
REQUIRED_COLUMNS = [
    'gauge_id', 'aridity_index', 'slope_mean', 'elev_mean', 
    'forest_frac', 'temp_mean', 'precip_mean', 'region', 'climate_class'
]

# Source datasets in Caravan
CARAVAN_SOURCES = ['camels', 'camelsaus', 'camelsbr', 'camelscl', 'camelsgb', 'hysets', 'lamah']

def download_file_with_progress(url: str, filename: str = None) -> bytes:
    """Download a file with progress bar."""
    logger.info(f"Downloading from: {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    if filename:
        logger.info(f"Saving to: {filename}")
    
    data = io.BytesIO()
    
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                data.write(chunk)
                pbar.update(len(chunk))
    
    return data.getvalue()

def extract_attributes_from_zip(zip_data: bytes) -> pd.DataFrame:
    """Extract and combine attributes from all sources in the zip file."""
    logger.info("Extracting attributes from zip file...")
    
    all_attributes = []
    
    with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
        # List all files in the zip
        file_list = zip_ref.namelist()
        
        for source in CARAVAN_SOURCES:
            # Look for both types of attribute files
            caravan_attr_file = f"attributes/attributes_caravan_{source}.csv"
            hydroatlas_attr_file = f"attributes/attributes_hydroatlas_{source}.csv"
            
            caravan_data = None
            hydroatlas_data = None
            
            # Find the actual file paths (they might have a prefix)
            caravan_files = [f for f in file_list if f.endswith(caravan_attr_file)]
            hydroatlas_files = [f for f in file_list if f.endswith(hydroatlas_attr_file)]
            
            if caravan_files and hydroatlas_files:
                logger.info(f"Processing {source} attributes...")
                
                # Read caravan attributes (climate indices)
                with zip_ref.open(caravan_files[0]) as f:
                    caravan_data = pd.read_csv(f)
                
                # Read hydroatlas attributes  
                with zip_ref.open(hydroatlas_files[0]) as f:
                    hydroatlas_data = pd.read_csv(f)
                
                # Merge on gauge_id
                if 'gauge_id' in caravan_data.columns and 'gauge_id' in hydroatlas_data.columns:
                    merged_data = pd.merge(caravan_data, hydroatlas_data, on='gauge_id', how='inner')
                    
                    # Add source column
                    merged_data['source'] = source
                    
                    # Filter to required columns (keep only those that exist)
                    available_cols = [col for col in REQUIRED_COLUMNS if col in merged_data.columns]
                    if 'source' not in available_cols:
                        available_cols.append('source')
                    
                    filtered_data = merged_data[available_cols]
                    all_attributes.append(filtered_data)
                    
                    logger.info(f"Added {len(filtered_data)} basins from {source}")
            else:
                logger.warning(f"Could not find attribute files for {source}")
    
    if not all_attributes:
        raise ValueError("No attribute data found in zip file")
    
    # Combine all sources
    combined_attributes = pd.concat(all_attributes, ignore_index=True)
    logger.info(f"Total basins loaded: {len(combined_attributes)}")
    
    return combined_attributes

def sample_basins(attributes_df: pd.DataFrame, n_basins: int = 100) -> List[str]:
    """Randomly sample basins ensuring good geographic/climatic diversity."""
    logger.info(f"Sampling {n_basins} basins...")
    
    # Remove basins with missing critical data
    clean_df = attributes_df.dropna(subset=['gauge_id'])
    
    if len(clean_df) < n_basins:
        logger.warning(f"Only {len(clean_df)} basins available, using all of them")
        n_basins = len(clean_df)
    
    # Try to sample diverse basins if we have the required columns
    if 'region' in clean_df.columns and len(clean_df['region'].unique()) > 1:
        # Sample proportionally from different regions
        sampled_basins = []
        regions = clean_df['region'].unique()
        
        basins_per_region = max(1, n_basins // len(regions))
        
        for region in regions:
            region_basins = clean_df[clean_df['region'] == region]
            sample_size = min(basins_per_region, len(region_basins))
            
            if sample_size > 0:
                regional_sample = region_basins.sample(n=sample_size, random_state=42)
                sampled_basins.append(regional_sample)
        
        # If we need more basins, sample from remaining
        sampled_df = pd.concat(sampled_basins, ignore_index=True)
        if len(sampled_df) < n_basins:
            remaining_basins = clean_df[~clean_df['gauge_id'].isin(sampled_df['gauge_id'])]
            additional_needed = n_basins - len(sampled_df)
            additional_sample = remaining_basins.sample(n=min(additional_needed, len(remaining_basins)), random_state=42)
            sampled_df = pd.concat([sampled_df, additional_sample], ignore_index=True)
        
        # Trim to exactly n_basins if we went over
        if len(sampled_df) > n_basins:
            sampled_df = sampled_df.sample(n=n_basins, random_state=42)
            
    else:
        # Simple random sampling
        sampled_df = clean_df.sample(n=n_basins, random_state=42)
    
    basin_ids = sampled_df['gauge_id'].tolist()
    logger.info(f"Selected {len(basin_ids)} basins")
    
    return basin_ids, sampled_df

def download_timeseries_for_basins(basin_ids: List[str], zip_data: bytes) -> pd.DataFrame:
    """Extract time series data for specified basins from the zip file."""
    logger.info(f"Extracting time series for {len(basin_ids)} basins...")
    
    all_timeseries = []
    
    with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
        file_list = zip_ref.namelist()
        
        for basin_id in tqdm(basin_ids, desc="Processing basins"):
            # Extract source and ID from gauge_id (format: {source}_{id})
            parts = basin_id.split('_', 1)
            if len(parts) != 2:
                logger.warning(f"Invalid gauge_id format: {basin_id}")
                continue
            
            source, original_id = parts
            
            # Look for the corresponding CSV file in timeseries/csv/{source}/
            pattern = f"timeseries/csv/{source}/"
            matching_files = [f for f in file_list if pattern in f and f.endswith('.csv')]
            
            # Find the specific basin file
            basin_file = None
            for file_path in matching_files:
                # Extract filename and check if it matches our basin
                filename = os.path.basename(file_path)
                if filename.startswith(original_id) or basin_id in filename:
                    basin_file = file_path
                    break
            
            if basin_file:
                try:
                    with zip_ref.open(basin_file) as f:
                        ts_data = pd.read_csv(f)
                    
                    # Add basin identifier
                    ts_data['gauge_id'] = basin_id
                    
                    # Keep only essential columns for the subset
                    essential_cols = ['gauge_id']
                    if 'date' in ts_data.columns:
                        essential_cols.append('date')
                    
                    # Add precipitation, temperature, and streamflow columns if they exist
                    precip_cols = [col for col in ts_data.columns if 'precipitation' in col.lower() or col.startswith('total_precipitation')]
                    temp_cols = [col for col in ts_data.columns if 'temperature' in col.lower() and '2m' in col]
                    flow_cols = [col for col in ts_data.columns if 'streamflow' in col.lower()]
                    
                    # Take mean values for temperature and precipitation, mean streamflow
                    if precip_cols:
                        essential_cols.extend([col for col in precip_cols if 'mean' in col or col == 'total_precipitation'])[:1]
                    if temp_cols:
                        essential_cols.extend([col for col in temp_cols if 'mean' in col])[:1]
                    if flow_cols:
                        essential_cols.extend([col for col in flow_cols if 'mean' in col or col == 'streamflow'])[:1]
                    
                    # Filter to available columns
                    available_cols = [col for col in essential_cols if col in ts_data.columns]
                    if available_cols:
                        filtered_ts = ts_data[available_cols]
                        all_timeseries.append(filtered_ts)
                        
                except Exception as e:
                    logger.warning(f"Error reading timeseries for {basin_id}: {e}")
            else:
                logger.warning(f"Timeseries file not found for basin: {basin_id}")
    
    if not all_timeseries:
        logger.error("No timeseries data found for any basin")
        return pd.DataFrame()
    
    # Combine all timeseries
    combined_timeseries = pd.concat(all_timeseries, ignore_index=True)
    logger.info(f"Combined timeseries shape: {combined_timeseries.shape}")
    
    return combined_timeseries

def main():
    """Main function to fetch Caravan subset."""
    logger.info("Starting Caravan subset download...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Get the download URL for Caravan.zip
    logger.info("Getting download URL from Zenodo API...")
    response = requests.get(ZENODO_FILES_URL)
    response.raise_for_status()
    
    files_info = response.json()
    caravan_file = None
    
    # Check if response has 'entries' field (Zenodo API structure)
    files_list = files_info.get('entries', [])
    
    for file_info in files_list:
        if file_info.get('key') == 'Caravan.zip':
            caravan_file = file_info
            break
    
    if not caravan_file:
        raise ValueError("Caravan.zip not found in Zenodo record")
    
    download_url = caravan_file['links']['content']
    file_size_gb = caravan_file['size'] / (1024**3)
    
    logger.info(f"Caravan.zip size: {file_size_gb:.1f} GB")
    logger.info("Note: We'll download the full zip but only extract needed parts")
    
    # Download the zip file
    zip_data = download_file_with_progress(download_url)
    
    # Extract and process attributes
    logger.info("Processing attributes...")
    attributes_df = extract_attributes_from_zip(zip_data)
    
    # Sample 100 basins
    basin_ids, sampled_attributes = sample_basins(attributes_df, n_basins=100)
    
    # Save attributes subset
    attributes_path = data_dir / "attributes.parquet"
    sampled_attributes.to_parquet(attributes_path, index=False)
    logger.info(f"Saved attributes to: {attributes_path}")
    
    # Save basin IDs
    basin_ids_path = data_dir / "demo_gauges.txt"
    with open(basin_ids_path, 'w') as f:
        for basin_id in basin_ids:
            f.write(f"{basin_id}\n")
    logger.info(f"Saved {len(basin_ids)} basin IDs to: {basin_ids_path}")
    
    # Download and save timeseries
    logger.info("Processing timeseries...")
    timeseries_df = download_timeseries_for_basins(basin_ids, zip_data)
    
    if not timeseries_df.empty:
        timeseries_path = data_dir / "timeseries.parquet" 
        timeseries_df.to_parquet(timeseries_path, index=False)
        logger.info(f"Saved timeseries to: {timeseries_path}")
        logger.info(f"Timeseries shape: {timeseries_df.shape}")
    else:
        logger.error("No timeseries data was extracted")
    
    logger.info("Caravan subset download completed!")
    
    # Print summary
    print("\n" + "="*50)
    print("CARAVAN SUBSET DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Attributes file: {attributes_path}")
    print(f"Basin IDs file: {basin_ids_path}")
    print(f"Timeseries file: {timeseries_path}")
    print(f"Number of basins: {len(basin_ids)}")
    print(f"Attributes shape: {sampled_attributes.shape}")
    if not timeseries_df.empty:
        print(f"Timeseries shape: {timeseries_df.shape}")
        print(f"Columns in attributes: {list(sampled_attributes.columns)}")
        print(f"Columns in timeseries: {list(timeseries_df.columns)}")
    print("="*50)

if __name__ == "__main__":
    main()