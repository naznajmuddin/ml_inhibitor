# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning system for predicting corrosion parameters of green plant-based inhibitors in acidic environments (H2SO4, HCl) on mild/carbon steel. The system implements gradient boosting models based on recent literature (Akrom et al. 2023, Ma et al. 2023).

### Target Inhibitors
- **Curry leaf extract** (Murraya koenigii)
- **Peanut shell extract** (Arachis hypogaea)
- **Aloe vera extract** (Aloe vera gel)

### Prediction Targets
- **IE%** - Inhibition efficiency (0-100%)
- **Ecorr** - Corrosion potential (mV vs SCE)
- **Icorr** - Corrosion current density (uA/cm2)
- **Weight loss** - Corrosion rate (various units)

### Supported Acids
- H2SO4 (sulfuric acid)
- HCl (hydrochloric acid)

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the complete pipeline
```bash
# Step 1: Preprocess data (filters to target inhibitors, adds engineered features, creates train/val/test splits)
python 01_data_preprocessing.py

# Step 2: Train ML models (trains 5 models + ensemble, generates visualizations)
python 02_ml_training_enhanced.py

# Step 3: Make predictions
python 03_predict.py                                                    # Interactive mode
python 03_predict.py --mode batch --input data.csv --output results.csv # Batch mode
python 03_predict.py --mode response_curve --inhibitor "Curry leaf extract" --output curve.png

# Step 4: Generate polarization curves (Tafel plots)
python 04_polarization_curves.py --inhibitor "Aloe vera extract" --output tafel_plot.png
```

## Architecture

### Four-Script Pipeline

1. **01_data_preprocessing.py** - Data preparation
   - Filters to target inhibitors: Curry leaf, Peanut shell, Aloe vera
   - Supports multiple acids: H2SO4, HCl
   - Adds engineered features: `log_conc_mg_L`, `temp_conc_interaction`, `acid_strength_norm`, `ln_Kads`
   - Uses `GroupShuffleSplit` by `paper_id` to prevent data leakage between train/val/test sets
   - Outputs: `preprocessed_data/` directory with CSVs and reports

2. **02_ml_training_enhanced.py** - Model training
   - Trains separate models for each target: IE%, Ecorr, Icorr
   - Uses 5 base models: GradientBoosting, HistGradientBoosting, RandomForest, Ridge, SVR
   - Creates VotingRegressor ensemble from top 3 models
   - Uses `GroupKFold` cross-validation grouped by `paper_id`
   - Feature preprocessing: `StandardScaler` for numeric, `OneHotEncoder` for categorical
   - Outputs: `ml_models/` (pickled models), `ml_figures/` (visualizations)

3. **03_predict.py** - Prediction tool
   - Three modes: interactive, batch, response_curve
   - Predicts multiple targets: IE%, Ecorr, Icorr
   - Loads trained models and preprocessor
   - Automatically adds engineered features before prediction

4. **04_polarization_curves.py** - Tafel curve generator (NEW)
   - Generates simulated Tafel polarization curves
   - Uses predicted/measured Ecorr, Icorr, ba, bc values
   - Visualizes blank vs inhibited steel comparison

### Key Feature Columns

**Numeric features:**
- `acid_molarity_M`, `temperature_C`, `immersion_time_h`, `inhibitor_conc_mg_L`
- `log_conc_mg_L`, `temp_conc_interaction`, `acid_strength_norm`

**Categorical features:**
- `acid`, `steel_grade`, `method`, `inhibitor_name`

**Targets:**
- `inhibition_efficiency_pct` (0-100%)
- `Ecorr_mV` (mV vs SCE, typically -400 to -600 for steel in acid)
- `Icorr_uA_cm2` (uA/cm2)
- `ba_mV_dec`, `bc_mV_dec` (Tafel slopes for polarization curves)

### Data Flow

```
Raw CSV --> 01_preprocessing --> preprocessed_data/*.csv --> 02_training --> ml_models/*.pkl --> 03_predict
                                                                      |
                                                                      v
                                                         04_polarization_curves (Tafel plots)
```

## Data Files

### Primary Data File
- `corrosion_inhibitors_expanded_v2.csv` - Main dataset with expanded schema

### CSV Schema (v2)
```
paper_id, inhibitor_name, inhibitor_scientific, steel_grade, acid, acid_molarity_M,
temperature_C, immersion_time_h, inhibitor_conc_mg_L, method, corrosion_rate_value,
corrosion_rate_unit, inhibition_efficiency_pct, Ecorr_mV, Icorr_uA_cm2, ba_mV_dec,
bc_mV_dec, notes
```

## Domain-Specific Notes

- Model supports both H2SO4 and HCl acidic environments
- Valid steel grades: ASTM A36, mild steel, carbon steel, Q235
- Concentration range: 0-3000 mg/L (extrapolation beyond this is unreliable)
- Temperature range: 25-70C is most reliable
- Predictions should always report uncertainty (typically +/-15% for IE%)
- `ln_Kads` is an alternative target for adsorption modeling (only valid for IE < 95%)

### Electrochemical Parameters
- **Ecorr**: More negative values indicate higher corrosion tendency
- **Icorr**: Lower values indicate better inhibition (IE% = (Icorr_blank - Icorr_inh)/Icorr_blank * 100)
- **ba, bc**: Tafel slopes used for polarization curve generation

## Data Sources

1. **Curry leaf**: CURRY_LEAVE.pdf - H2SO4, ASTM A36, Weight loss + PDP
2. **Peanut shell**: main.pdf - H2SO4, Q235 carbon steel, Weight loss
3. **Aloe vera**: Singh_2016_AloeVera.pdf - HCl, mild steel, Weight loss + EIS + PDP
   - Source: Singh et al. (2016) J. Ind. Eng. Chem. 33:288-297
