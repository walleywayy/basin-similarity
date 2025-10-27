# Basin Similarity and Drought Prediction

A comprehensive Python framework for analyzing basin characteristics, predicting drought probability, and mapping basin similarity using hydrological datasets.

## 🌊 Overview

This project provides tools to:
- **Download and process** hydrological datasets (CAMELS-CH, Caravan)
- **Analyze basin characteristics** and compute drought indicators
- **Predict drought probability** for unseen basins based on their characteristics
- **Map basin similarity** using dimensionality reduction techniques
- **Interactive prediction** interface for real-time basin analysis

## 📁 Project Structure

```
basin-similarity/
├── data/                           # Data files
│   ├── attributes.csv              # Basin characteristics
│   ├── timeseries.csv              # Streamflow and meteorological data
│   └── demo_gauges.txt             # Basin IDs for analysis
├── scripts/                        # Main analysis scripts
│   ├── fetch_caravan_subset_optimized.py  # Data download
│   ├── basic_drought_analysis.py         # Exploratory analysis
│   ├── predict_droughts.py               # Full ML pipeline
│   ├── predict_unseen_basin.py           # Prediction for new basins
│   ├── interactive_basin_prediction.py   # Interactive interface
│   └── create_synthetic_data.py         # Generate test data
├── notebooks/                      # Jupyter notebooks
│   └── predict_droughts.ipynb     # Analysis notebook
├── models/                         # Trained models (generated)
├── requirements.txt                # Python dependencies
├── Makefile                       # Build commands
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/basin-similarity.git
cd basin-similarity

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Download Swiss CAMELS-CH dataset (fast, ~250MB)
python scripts/fetch_caravan_subset_optimized.py --dataset camels_ch --n-basins 50

# Or download global Caravan dataset (slower, ~12GB)
python scripts/fetch_caravan_subset_optimized.py --dataset caravan --n-basins 50 --size-cap-mb 500
```

### 3. Run Analysis

```bash
# Basic exploratory analysis
python scripts/basic_drought_analysis.py

# Full machine learning pipeline
python scripts/predict_droughts.py

# Interactive basin prediction
python scripts/interactive_basin_prediction.py
```

## 📊 Features

### Data Processing
- **Memory-efficient** ZIP file processing
- **Automatic column detection** and standardization
- **Missing data handling** with intelligent imputation
- **Multiple dataset support** (CAMELS-CH, Caravan)

### Drought Analysis
- **SRI calculation** (Standardized Runoff Index)
- **Monthly aggregation** with drought flagging
- **Meteorological feature engineering** (lagged precipitation, temperature)
- **Basin similarity mapping** using PCA

### Machine Learning
- **Random Forest models** for drought classification and streamflow prediction
- **Feature importance analysis**
- **Model persistence** with joblib
- **Cross-validation** and performance metrics

### Interactive Interface
- **Real-time prediction** based on basin characteristics
- **Similar basin finding** using reference datasets
- **Result export** to JSON format
- **User-friendly** command-line interface

## 🔧 Usage Examples

### Interactive Basin Prediction

```bash
python scripts/interactive_basin_prediction.py
```

Example input:
```
aridity_index (0-2, higher = more arid) [default: 1.0]: 1.5
precip_mean (mm/year) [default: 800]: 400
temp_mean (°C) [default: 10]: 15
elev_mean (meters) [default: 500]: 1200
forest_frac (0-1) [default: 0.3]: 0.2
```

Output:
```
Drought Probability: 35.2%
Predicted Streamflow: 3.45 m³/s
Classification: MODERATE DROUGHT RISK
```

### Programmatic Prediction

```python
from scripts.predict_unseen_basin import predict_basin

# Define basin characteristics
basin_attrs = {
    'aridity_index': 1.2,
    'precip_mean': 600,
    'temp_mean': 12,
    'elev_mean': 800,
    'forest_frac': 0.4
}

# Make prediction
results = predict_basin(basin_attrs)
print(f"Drought probability: {results['drought_probability']:.1%}")
```

### Batch Processing

```bash
# Create example basin JSON
python scripts/predict_unseen_basin.py --save-example

# Predict from JSON file
python scripts/predict_unseen_basin.py --json-file example_basin.json
```

## 📈 Key Metrics

The system predicts:
- **Drought Probability**: 0-100% based on basin characteristics
- **Streamflow**: Average monthly streamflow in m³/s
- **Basin Similarity**: PCA coordinates for similarity mapping
- **Risk Classification**: LOW/MODERATE/HIGH drought risk

## 🎯 Model Performance

Typical performance metrics:
- **Drought Classification**: 75-85% accuracy
- **Streamflow Prediction**: R² = 0.6-0.8
- **Feature Importance**: Precipitation, temperature, aridity index

## 🔬 Scientific Background

### Drought Indicators
- **SRI (Standardized Runoff Index)**: Basin-specific streamflow standardization
- **Threshold**: SRI < -1.0 indicates drought conditions
- **Temporal Resolution**: Monthly aggregation

### Basin Similarity
- **PCA Dimensionality Reduction**: Maps high-dimensional basin characteristics to 2D
- **Feature Engineering**: Lagged meteorological variables
- **Similarity Metrics**: Euclidean distance in PCA space

### Prediction Features
- **Physical**: Elevation, slope, forest fraction
- **Climatic**: Precipitation, temperature, aridity index
- **Hydrological**: Streamflow patterns, drought history

## 🛠️ Development

### Adding New Datasets

1. Update `DATASET_CONFIGS` in `fetch_caravan_subset_optimized.py`
2. Add dataset-specific parsing logic
3. Update column mappings in `pick_and_rename_timeseries_cols()`

### Extending Models

1. Modify feature engineering in `predict_droughts.py`
2. Add new model types (XGBoost, Neural Networks)
3. Update prediction logic in `predict_unseen_basin.py`

### Custom Analysis

1. Create new scripts in `scripts/` directory
2. Use existing data loading functions
3. Follow the established analysis pipeline

## 📋 Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- scikit-learn >= 1.0.0
- requests >= 2.25.0
- tqdm >= 4.60.0
- joblib >= 1.0.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CAMELS-CH**: Swiss hydrological dataset
- **Caravan**: Global hydrological dataset
- **Zenodo**: Data hosting platform
- **scikit-learn**: Machine learning library

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact: [your-email@domain.com]
- Documentation: [project-wiki-url]

---

**Happy analyzing! 🌊📊**