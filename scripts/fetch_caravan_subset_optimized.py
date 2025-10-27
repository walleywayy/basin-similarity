#!/usr/bin/env python3
"""
Optimized memory-efficient script to fetch hydrology dataset subsets.
Reads directly from ZIP files without bulk extraction.
"""

import argparse
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Zenodo API URLs
CARAVAN_RECORD = "7540792"
CAMELS_CH_RECORD = "15025258"
ZENODO_BASE_URL = "https://zenodo.org/api/records"

# Dataset configurations
DATASET_CONFIGS = {
    'caravan': {
        'record_id': CARAVAN_RECORD,
        'zip_file': 'Caravan.zip',
        'size_gb': 12.5,
        'sources': ['camels', 'camelsaus', 'camelsbr', 'camelscl', 'camelsgb', 'hysets', 'lamah']
    },
    'camels_ch': {
        'record_id': CAMELS_CH_RECORD,
        'zip_file': 'camels_ch.zip',
        'size_gb': 0.247,
        'sources': ['camels_ch']
    }
}

# Required columns from attributes files
REQUIRED_COLUMNS = [
    'gauge_id', 'aridity_index', 'slope_mean', 'elev_mean', 
    'forest_frac', 'temp_mean', 'precip_mean', 'region', 'climate_class'
]

def stream_zip_to_temp(url: str, label: str, chunk_mb: int = 4) -> Path:
    """Stream ZIP download to temporary file."""
    logger.info(f"Streaming download from: {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    tmp = Path(tempfile.mkstemp(suffix=".zip")[1])
    with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {label}") as pbar:
        for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    logger.info(f"Saved temp zip: {tmp}")
    return tmp

def read_attributes_from_zip(zip_path: Path, dataset: str) -> pd.DataFrame:
    """Read attributes directly from ZIP file."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        frames = []

        if dataset == "camels_ch":
            # Heuristic: look for attribute tables
            cand = [n for n in names if n.lower().endswith(".csv") and any(k in n.lower() for k in ["attr", "attributes", "catchment"])]
            if not cand:
                # fallback: list all CSVs and pick the largest in camels_ch root
                cand = [n for n in names if n.lower().endswith(".csv")]
                cand.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
                cand = cand[:1]
            for n in cand:
                try:
                    df = pd.read_csv(zf.open(n), encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(zf.open(n), encoding='latin-1')
                    except UnicodeDecodeError:
                        df = pd.read_csv(zf.open(n), encoding='cp1252')
                df["source"] = "camels_ch"
                frames.append(df)
        else:
            # Caravan: merge caravan + hydroatlas per source
            for src in DATASET_CONFIGS["caravan"]["sources"]:
                car = [n for n in names if n.endswith(f"attributes/attributes_caravan_{src}.csv")]
                hyd = [n for n in names if n.endswith(f"attributes/attributes_hydroatlas_{src}.csv")]
                if not (car and hyd):
                    continue
                try:
                    car_df = pd.read_csv(zf.open(car[0]), encoding='utf-8')
                except UnicodeDecodeError:
                    car_df = pd.read_csv(zf.open(car[0]), encoding='latin-1')
                try:
                    hyd_df = pd.read_csv(zf.open(hyd[0]), encoding='utf-8')
                except UnicodeDecodeError:
                    hyd_df = pd.read_csv(zf.open(hyd[0]), encoding='latin-1')
                if "gauge_id" not in car_df or "gauge_id" not in hyd_df:
                    continue
                m = pd.merge(car_df, hyd_df, on="gauge_id", how="inner")
                m["source"] = src
                frames.append(m)

        if not frames:
            raise RuntimeError("No attribute tables found in the ZIP.")

        attrs = pd.concat(frames, ignore_index=True)
        keep = [c for c in REQUIRED_COLUMNS if c in attrs.columns] + ["source"]
        return attrs[keep]

def sample_basins(attributes_df: pd.DataFrame, dataset: str, n_basins: int = 100) -> Tuple[List[str], pd.DataFrame]:
    """Randomly sample basins ensuring good geographic/climatic diversity."""
    df = attributes_df.copy()

    # Try to find a numeric id column for CAMELS-CH; otherwise fall back to existing gauge_id
    id_col = None
    for c in ["gauge_id", "gaugeid", "station_id", "basin_id", "id"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        df[id_col := "id"] = np.arange(len(df))  # fallback

    # Sample
    if "region" in df and df["region"].nunique() > 1:
        per = max(1, n_basins // df["region"].nunique())
        parts = []
        for r, g in df.groupby("region"):
            parts.append(g.sample(n=min(per, len(g)), random_state=42))
        samp = pd.concat(parts, ignore_index=True)
        if len(samp) > n_basins: 
            samp = samp.sample(n=n_basins, random_state=42)
        if len(samp) < n_basins:
            rest = df[~df.index.isin(samp.index)]
            need = min(n_basins - len(samp), len(rest))
            samp = pd.concat([samp, rest.sample(n=need, random_state=42)], ignore_index=True)
    else:
        samp = df.sample(n=min(n_basins, len(df)), random_state=42)

    # Build a clean gauge_id
    if dataset == "camels_ch":
        orig = samp[id_col].astype(str).str.replace(r"\D+", "", regex=True)  # keep digits
        samp["gauge_id"] = "camels_ch_" + orig
        samp["orig_id"] = orig
    else:
        # assume already like 'source_xxx'; ensure string
        samp["gauge_id"] = samp["gauge_id"].astype(str)
        samp["orig_id"] = samp["gauge_id"].str.split("_", 1).str[-1]

    return samp["gauge_id"].tolist(), samp

def pick_and_rename_timeseries_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Pick and rename timeseries columns for consistency."""
    # date
    d = [c for c in df.columns if c.lower() in ["date","time","datetime"]]
    # flow
    q = [c for c in df.columns if any(k in c.lower() for k in ["q","qobs","discharge","stream","flow","runoff"])]
    # precip/temp (optional)
    p = [c for c in df.columns if any(k in c.lower() for k in ["precip","prcp","ppt","total_precipitation"])]
    t = [c for c in df.columns if any(k in c.lower() for k in ["t_mean","tavg","temperature","temp","t2m","2m"])]

    keep = []
    rename = {}
    if d: keep.append(d[0]); rename[d[0]] = "date"
    if q: keep.append(q[0]); rename[q[0]] = "streamflow"
    if p: keep.append(p[0]); rename[p[0]] = "precipitation"
    if t: keep.append(t[0]); rename[t[0]] = "temperature"

    out = df[keep].rename(columns=rename)
    return out

def extract_timeseries_from_zip(zip_path: Path, basin_ids: List[str], dataset: str, out_csv: Path, size_cap_mb: int = None) -> float:
    """Extract timeseries from ZIP and write to CSV incrementally."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        total_mb = 0
        wrote_header = False

        def find_camels_ch_file(orig_id: str) -> str:
            # Try daily first; fall back to annual if needed
            cand = [n for n in names if n.endswith(".csv") and f"/{orig_id}.csv" in n]
            # also allow filenames that start with id
            if not cand:
                cand = [n for n in names if n.endswith(".csv") and os.path.basename(n).startswith(orig_id)]
            return cand[0] if cand else ""

        def find_caravan_file(gid: str) -> str:
            src, orig = gid.split("_", 1)
            cand = [n for n in names if n.startswith("timeseries/csv/") and f"/{src}/" in n and n.endswith(".csv")]
            for p in cand:
                base = os.path.basename(p)
                if base == f"{orig}.csv" or base.startswith(orig):
                    return p
            return ""

        # fresh output file
        if out_csv.exists(): 
            out_csv.unlink()

        for gid in tqdm(basin_ids, desc="Timeseries"):
            if dataset == "camels_ch":
                orig = gid.split("_", 2)[-1]
                path = find_camels_ch_file(orig)
            else:
                path = find_caravan_file(gid)

            if not path:
                logger.warning(f"No timeseries for {gid}")
                continue

            # size guard (approx)
            total_mb += zf.getinfo(path).file_size / (1024**2)
            if size_cap_mb and total_mb > size_cap_mb:
                logger.warning(f"Size cap {size_cap_mb} MB reached, stopping.")
                break

            try:
                df = pd.read_csv(zf.open(path), encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(zf.open(path), encoding='latin-1')
                except UnicodeDecodeError:
                    df = pd.read_csv(zf.open(path), encoding='cp1252')
            df = pick_and_rename_timeseries_cols(df)
            if "date" not in df or "streamflow" not in df:
                logger.warning(f"Missing date/streamflow in {path}, skipping")
                continue
            df.insert(0, "gauge_id", gid)

            # Downcast numerics
            for c in df.columns:
                if c == "gauge_id" or c == "date": 
                    continue
                if pd.api.types.is_float_dtype(df[c]): 
                    df[c] = df[c].astype("float32")
                if pd.api.types.is_integer_dtype(df[c]): 
                    df[c] = df[c].astype("Int32")

            # append
            df.to_csv(out_csv, index=False, mode="a", header=not wrote_header)
            wrote_header = True

        return total_mb

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fetch a subset of hydrology dataset")
    parser.add_argument('--dataset', choices=['camels_ch', 'caravan'], default='camels_ch',
                       help='Dataset to use: camels_ch (247MB, fast) or caravan (12.5GB, slow)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only process attributes, skip timeseries download')
    parser.add_argument('--n-basins', type=int, default=100,
                       help='Number of basins to sample (default: 100)')
    parser.add_argument('--size-cap-mb', type=int, default=None,
                       help='Maximum size in MB for timeseries data (default: no limit)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory (default: data)')
    return parser.parse_args()

def main():
    """Main function to fetch dataset subset."""
    args = parse_arguments()

    logger.info("Starting dataset subset download...")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Number of basins: {args.n_basins}")
    logger.info(f"Size cap: {args.size_cap_mb}MB" if args.size_cap_mb else "No size cap")

    # Get dataset configuration
    config = DATASET_CONFIGS[args.dataset]
    zenodo_files_url = f"{ZENODO_BASE_URL}/{config['record_id']}/files"

    # Create output directory
    data_dir = Path(args.output_dir)
    data_dir.mkdir(exist_ok=True)

    # Check for cached attributes in dry-run mode
    attributes_path = data_dir / "attributes.csv"
    if args.dry_run and attributes_path.exists():
        logger.info("Using cached attributes for dry run...")
        sampled_attributes = pd.read_csv(attributes_path)
        basin_ids = sampled_attributes['gauge_id'].tolist()
        logger.info(f"Loaded {len(basin_ids)} basins from cache")
    else:
        # Get the download URL for the dataset zip file
        logger.info(f"Getting download URL from Zenodo API for {args.dataset}...")
        response = requests.get(zenodo_files_url)
        response.raise_for_status()

        files_info = response.json()
        dataset_file = None
        
        # Check both possible API structures
        files_list = files_info.get('entries', files_info.get('files', []))
        
        for file_info in files_list:
            if file_info.get('key') == config['zip_file']:
                dataset_file = file_info
                break

        if not dataset_file:
            raise ValueError(f"{config['zip_file']} not found in Zenodo record")

        download_url = dataset_file['links']['content']
        file_size_gb = dataset_file['size'] / (1024**3)

        logger.info(f"{config['zip_file']} size: {file_size_gb:.1f} GB")

        # Download ZIP to temp file
        zip_path = stream_zip_to_temp(download_url, config['zip_file'])
        
        try:
            # Read attributes directly from ZIP
            attrs = read_attributes_from_zip(zip_path, args.dataset)
            basin_ids, sampled_attributes = sample_basins(attrs, args.dataset, n_basins=args.n_basins)

            # Save attributes + ids
            sampled_attributes.to_csv(attributes_path, index=False)
            with open(data_dir / "demo_gauges.txt", "w") as f:
                for gid in basin_ids:
                    f.write(gid + "\n")

            if not args.dry_run:
                ts_path = data_dir / "timeseries.csv"
                size_mb = extract_timeseries_from_zip(zip_path, basin_ids, args.dataset, ts_path, args.size_cap_mb)
                logger.info(f"Timeseries written (~{size_mb:.1f} MB) -> {ts_path}")
            else:
                logger.info("Dry run: skipped timeseries extraction.")

        finally:
            try:
                os.remove(zip_path)
                logger.info("Removed temp zip.")
            except Exception:
                pass

    logger.info("Dataset subset download completed!")

    # Print summary
    print("\n" + "="*50)
    print(f"{args.dataset.upper()} SUBSET DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Attributes file: {attributes_path}")
    print(f"Basin IDs file: {data_dir / 'demo_gauges.txt'}")
    if not args.dry_run:
        ts_path = data_dir / "timeseries.csv"
        if ts_path.exists():
            print(f"Timeseries file: {ts_path}")
    print(f"Number of basins: {len(basin_ids)}")
    print(f"Attributes shape: {sampled_attributes.shape}")
    if not args.dry_run:
        ts_path = data_dir / "timeseries.csv"
        if ts_path.exists():
            ts_df = pd.read_csv(ts_path)
            print(f"Timeseries shape: {ts_df.shape}")
            print(f"Columns in timeseries: {list(ts_df.columns)}")
    print("="*50)

if __name__ == "__main__":
    main()
