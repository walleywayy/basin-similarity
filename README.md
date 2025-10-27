# Basin Similarity Analysis

A memory-efficient framework for analyzing streamflow patterns and drought prediction using global river basin datasets.

## 🎯 Project Overview

This project explores **basin similarity** — how rivers with similar climate, topography, and vegetation characteristics exhibit similar flow behavior. We use this similarity to predict streamflow and identify drought patterns across different basins.

### Key Features

- **Memory-efficient data processing**: Streams large datasets without loading everything into RAM
- **Multi-dataset support**: CAMELS-CH (247MB, fast) and Caravan (12.5GB, comprehensive)
- **Drought prediction**: Standardized Runoff Index (SRI) and machine learning models
- **Basin similarity mapping**: UMAP/PCA for visualizing basin relationships

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset Subset

**Fast prototype with CAMELS-CH (Switzerland):**
```bash
# Dry run - just get basin attributes
python scripts/fetch_caravan_subset_optimized.py --dataset camels_ch --dry-run --n-basins 50

# Full dataset with timeseries
python scripts/fetch_caravan_subset_optimized.py --dataset camels_ch --n-basins 80 --size-cap-mb 300
```

**Large Caravan dataset (global):**
```bash
# Dry run first
python scripts/fetch_caravan_subset_optimized.py --dataset caravan --dry-run --n-basins 100

# Full dataset (takes longer to download)
python scripts/fetch_caravan_subset_optimized.py --dataset caravan --n-basins 100 --size-cap-mb 500
```

### 3. Run Analysis

```bash
# Basic analysis (works with or without timeseries)
python scripts/basic_drought_analysis.py

# Advanced ML analysis (requires sklearn)
python scripts/predict_droughts.py
```

## 📊 Dataset Information

| Dataset | Size | Basins | Region | Download Time |
|---------|------|--------|--------|---------------|
| CAMELS-CH | 247MB | ~500 | Switzerland | ~4 minutes |
| Caravan | 12.5GB | ~6,000 | Global | ~2-4 hours |

## 🔧 Command Line Options

```bash
python scripts/fetch_caravan_subset_optimized.py [OPTIONS]

Options:
  --dataset {camels_ch,caravan}    Dataset to use (default: camels_ch)
  --dry-run                        Only download attributes, skip timeseries
  --n-basins N                     Number of basins to sample (default: 100)
  --size-cap-mb N                  Maximum timeseries size in MB (default: 300)
  --output-dir DIR                 Output directory (default: data/)
```

## 📁 Output Files

After running the pipeline, you'll get:

- `data/attributes.csv` - Basin characteristics (elevation, climate, etc.)
- `data/demo_gauges.txt` - Clean basin IDs (`camels_ch_350` format)
- `data/timeseries.csv` - Streamflow and meteorological data
- `drought_analysis.png` - Analysis visualizations

## 🧠 Analysis Framework

### 1. Data Processing
- **Streaming download**: Downloads ZIP files without loading into RAM
- **Direct ZIP reading**: Extracts only needed files without full decompression
- **Clean gauge IDs**: Normalizes basin identifiers (`camels_ch_350` format)
- **Robust encoding**: Handles Unicode issues in CSV files

### 2. Drought Analysis
- **Standardized Runoff Index (SRI)**: `(flow - mean) / std` per basin
- **Drought threshold**: SRI < -1.0 indicates drought conditions
- **Monthly aggregation**: Converts daily data to monthly means
- **Lagged features**: 3-month rolling averages for precipitation/temperature

### 3. Basin Similarity
- **Attribute correlation**: Identifies key basin characteristics
- **Flow pattern analysis**: Compares streamflow distributions
- **Drought frequency**: Maps drought susceptibility across basins
- **Similarity metrics**: Statistical measures for basin comparison

## 🔬 Technical Implementation

### Memory Optimization
```python
# Streams ZIP download to temp file
zip_path = stream_zip_to_temp(download_url, config['zip_file'])

# Reads directly from ZIP without extraction
with zipfile.ZipFile(zip_path, "r") as zf:
    df = pd.read_csv(zf.open(file_path))
```

### Gauge ID Normalization
```python
# Fixes problematic IDs like 'camels_ch,350' → 'camels_ch_350'
orig = samp[id_col].astype(str).str.replace(r"\D+", "", regex=True)
samp["gauge_id"] = "camels_ch_" + orig
```

### Drought Labeling
```python
# SRI-like standardization per basin
def sri(df):
    mu, sd = df["streamflow"].mean(), df["streamflow"].std()
    df["SRI"] = (df["streamflow"] - mu) / (sd if sd > 0 else 1.0)
    return df

monthly["drought_flag"] = (monthly["SRI"] < -1.0).astype(int)
```

## 📈 Example Results

### CAMELS-CH Dataset (50 basins)
```
Attributes shape: (50, 4)
Available attributes: ['source', 'id', 'gauge_id', 'orig_id']
Clean gauge IDs: ['camels_ch_350', 'camels_ch_3439', 'camels_ch_1020', ...]

Basin attribute summary:
                id      orig_id
count    50.000000    50.000000
mean   2131.860000  2131.860000
std    1344.252786  1344.252786
min      23.000000    23.000000
max    4428.000000  4428.000000
```

## 🛠️ Development

### Project Structure
```
basin-similarity/
├── scripts/
│   ├── fetch_caravan_subset_optimized.py  # Main data pipeline
│   ├── basic_drought_analysis.py         # Basic analysis
│   └── predict_droughts.py               # ML models
├── notebooks/
│   └── predict_droughts.ipynb            # Jupyter analysis
├── data/                                 # Output directory
└── requirements.txt                      # Dependencies
```

### Key Scripts

1. **`fetch_caravan_subset_optimized.py`**: Memory-efficient data pipeline
2. **`basic_drought_analysis.py`**: Drought analysis without complex dependencies
3. **`predict_droughts.py`**: Full ML pipeline with sklearn

## 🎯 Next Steps

1. **Install sklearn**: For advanced machine learning models
2. **Add UMAP**: For better basin similarity visualization
3. **Expand features**: Include more meteorological variables
4. **Interactive plots**: Create web-based visualizations
5. **Model validation**: Cross-validation and performance metrics

## 📚 References

- [CAMELS-CH Dataset](https://zenodo.org/record/15025258)
- [Caravan Dataset](https://zenodo.org/record/7540792)
- [Standardized Runoff Index](https://doi.org/10.1029/2004WR003509)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.