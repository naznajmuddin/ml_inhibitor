# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning system for predicting corrosion inhibition efficiency (IE%) of green plant-based inhibitors in H2SO4 environments on mild/carbon steel. The system implements gradient boosting models based on recent literature (Akrom et al. 2023, Ma et al. 2023).

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the complete pipeline
```bash
# Step 1: Preprocess data (filters to H2SO4 + mild steel, adds engineered features, creates train/val/test splits)
python 01_data_preprocessing.py

# Step 2: Train ML models (trains 5 models + ensemble, generates visualizations)
python 02_ml_training_enhanced.py

# Step 3: Make predictions
python 03_predict.py                                                    # Interactive mode
python 03_predict.py --mode batch --input data.csv --output results.csv # Batch mode
python 03_predict.py --mode response_curve --inhibitor "Curry leaf extract" --output curve.png
```

## Architecture

### Three-Script Pipeline

1. **01_data_preprocessing.py** - Data preparation
   - Filters to H2SO4 acid and mild/carbon steel only
   - Adds engineered features: `log_conc_mg_L`, `temp_conc_interaction`, `acid_strength_norm`, `ln_Kads`
   - Uses `GroupShuffleSplit` by `paper_id` to prevent data leakage between train/val/test sets
   - Outputs: `preprocessed_data/` directory with CSVs and reports

2. **02_ml_training_enhanced.py** - Model training
   - Trains 5 models: GradientBoosting, HistGradientBoosting, RandomForest, Ridge, SVR
   - Creates VotingRegressor ensemble from top 3 models
   - Uses `GroupKFold` cross-validation grouped by `paper_id`
   - Feature preprocessing: `StandardScaler` for numeric, `OneHotEncoder` for categorical
   - Outputs: `ml_models/` (pickled models), `ml_figures/` (visualizations)

3. **03_predict.py** - Prediction tool
   - Three modes: interactive, batch, response_curve
   - Loads `best_model_model.pkl` and `best_model_preprocessor.pkl`
   - Automatically adds engineered features before prediction

### Key Feature Columns

**Numeric features:**
- `acid_molarity_M`, `temperature_C`, `immersion_time_h`, `inhibitor_conc_mg_L`
- `log_conc_mg_L`, `temp_conc_interaction`, `acid_strength_norm`

**Categorical features:**
- `acid`, `steel_grade`, `method`, `inhibitor_name`

**Target:** `inhibition_efficiency_pct` (0-100%)

### Data Flow

```
Raw CSV → 01_preprocessing → preprocessed_data/*.csv → 02_training → ml_models/*.pkl → 03_predict
```

## Domain-Specific Notes

- Model is specifically trained for H2SO4 environment only (not HCl, HNO3, etc.)
- Valid steel grades: ASTM A36, mild steel, carbon steel, Q235
- Concentration range: 0-3000 mg/L (extrapolation beyond this is unreliable)
- Temperature range: 25-70°C is most reliable
- Predictions should always report uncertainty (typically ±15%)
- `ln_Kads` is an alternative target for adsorption modeling (only valid for IE < 95%)
