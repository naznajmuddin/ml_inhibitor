# Corrosion Inhibitor ML Prediction System

## 🎯 Overview

This system implements state-of-the-art machine learning for predicting corrosion inhibition efficiency (IE%) of green plant-based inhibitors in H2SO4 environments. Built on best practices from recent literature (2023-2025).

### Key Features
- ✅ Gradient Boosting models (Akrom et al. 2023 - best performer)
- ✅ Enhanced feature engineering (log concentration, interactions)
- ✅ Group-based cross-validation (prevents data leakage)
- ✅ Uncertainty quantification
- ✅ Concentration-response curve generation
- ✅ Publication-quality visualizations

---

## 📁 Project Structure

```
corrosion_inhibitor_ml/
├── corrosion_inhibitors_literature_expanded.csv  # Your raw data
├── requirements.txt                               # Python dependencies
├── 01_data_preprocessing.py                       # Data preparation
├── 02_ml_training_enhanced.py                     # Model training
├── 03_predict.py                                  # Prediction tool
├── preprocessed_data/                             # Processed datasets
│   ├── train_data.csv
│   ├── val_data.csv
│   ├── test_data.csv
│   └── ...
├── ml_models/                                     # Trained models
│   ├── best_model_model.pkl
│   ├── best_model_preprocessor.pkl
│   └── ...
└── ml_figures/                                    # Visualizations
    ├── feature_importance.png
    ├── predictions_vs_actual.png
    └── ...
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy, pandas, scipy
- scikit-learn, joblib
- matplotlib, seaborn

### 2. Prepare Your Data

**Option A: Use your existing dataset**

Ensure your CSV has these columns:
- `paper_id` - Source identifier
- `inhibitor_name` - Name of inhibitor
- `acid` - Type of acid (e.g., "H2SO4")
- `steel_grade` - Steel type
- `acid_molarity_M` - Acid concentration
- `temperature_C` - Temperature in Celsius
- `immersion_time_h` - Immersion time
- `inhibitor_conc_mg_L` - Inhibitor concentration (mg/L)
- `method` - Experimental method (e.g., "Weight loss")
- `inhibition_efficiency_pct` - Target variable (0-100%)

**Option B: Add more data from literature**

See `corrosion_inhibitor_research_sources.md` for recommended sources with high-performing inhibitors.

### 3. Run Data Preprocessing

```bash
python 01_data_preprocessing.py
```

**What it does:**
- Filters to H2SO4 and mild/carbon steel
- Adds engineered features (log concentration, interactions)
- Creates train/validation/test splits (grouped by paper to prevent leakage)
- Generates data quality report and visualizations

**Outputs:**
- `preprocessed_data/train_data.csv` - Training set
- `preprocessed_data/val_data.csv` - Validation set
- `preprocessed_data/test_data.csv` - Test set
- `preprocessed_data/PREPROCESSING_REPORT.txt` - Detailed report
- `preprocessed_data/data_distribution_analysis.png` - Visualizations

**Review the outputs** before proceeding to ensure data quality!

### 4. Train ML Models

```bash
python 02_ml_training_enhanced.py
```

**What it does:**
- Trains 5 different models (GradientBoosting, HistGradientBoosting, RandomForest, Ridge, SVR)
- Creates ensemble model for robustness
- Performs group-based cross-validation
- Generates comprehensive visualizations
- Selects best model based on validation performance

**Outputs:**
- `ml_models/best_model_model.pkl` - Trained model
- `ml_models/best_model_preprocessor.pkl` - Feature preprocessor
- `ml_models/TRAINING_REPORT.txt` - Full training report
- `ml_figures/feature_importance.png` - Feature importance
- `ml_figures/predictions_vs_actual_*.png` - Prediction quality
- `ml_figures/residuals_analysis.png` - Error analysis
- `ml_figures/concentration_response_curves.png` - Response curves

**Expected Performance** (based on literature):
- R² > 0.80 (good)
- R² > 0.85 (very good)
- R² > 0.90 (excellent - matches Akrom et al. 2023)

### 5. Make Predictions

**Interactive Mode (easiest):**

```bash
python 03_predict.py
```

Follow the prompts to enter experimental conditions and get instant predictions.

**Batch Predictions:**

```bash
python 03_predict.py --mode batch --input new_experiments.csv --output predictions.csv
```

**Generate Concentration-Response Curve:**

```bash
python 03_predict.py --mode response_curve --inhibitor "Curry leaf extract" --output curve.png
```

---

## 📊 Understanding the Results

### Feature Importance

The top features typically are:
1. **inhibitor_conc_mg_L** - Concentration (expected, as IE increases with concentration)
2. **log_conc_mg_L** - Log concentration (captures adsorption isotherm behavior)
3. **temperature_C** - Temperature (affects adsorption mechanism)
4. **temp_conc_interaction** - Temperature × Concentration (captures physisorption vs chemisorption)

### Model Performance Metrics

- **R² (R-squared)**: How well the model explains variance (0-1, higher is better)
  - R² > 0.90 = Excellent
  - R² > 0.85 = Very good
  - R² > 0.80 = Good
  - R² < 0.70 = Needs improvement

- **MAE (Mean Absolute Error)**: Average prediction error in percentage points
  - MAE < 10% = Excellent
  - MAE < 15% = Good
  - MAE < 20% = Acceptable
  - MAE > 20% = Needs improvement

- **RMSE (Root Mean Squared Error)**: Like MAE but penalizes large errors more

### Uncertainty

All predictions come with uncertainty. Typical model uncertainty is ±15-20% based on:
- Cross-validation standard deviation
- Test set MAE
- Ensemble variance

**Always report predictions as:** `Predicted IE: 85% ± 15%`

---

## 🔬 Advanced Usage

### Adding New Data

1. Add new rows to your CSV with the required columns
2. Re-run preprocessing: `python 01_data_preprocessing.py`
3. Re-train model: `python 02_ml_training_enhanced.py`

The model will automatically incorporate the new data!

### Tuning Hyperparameters

Edit the model definitions in `02_ml_training_enhanced.py`:

```python
"GradientBoosting": GradientBoostingRegressor(
    n_estimators=200,      # Increase for more trees (slower but potentially better)
    max_depth=4,           # Increase for more complex relationships
    min_samples_leaf=15,   # Decrease for less regularization
    learning_rate=0.05,    # Decrease for slower, more careful learning
    subsample=0.8,         # Fraction of samples per tree
    random_state=42,
),
```

### Using Alternative Targets

The preprocessing script calculates `ln_Kads` (logarithm of adsorption constant) as an alternative target. To use it:

1. In `02_ml_training_enhanced.py`, change:
```python
TARGET_IE = "inhibition_efficiency_pct"  # Current
TARGET_IE = "ln_Kads"  # Alternative (more thermodynamically grounded)
```

2. Filter data to samples where `ln_Kads` is available (IE < 95%)

This avoids saturation effects at high IE% values.

### Virtual Sample Generation (Advanced)

For small datasets, you can augment with virtual samples using KDE:

```python
from scipy.stats import gaussian_kde

# In 01_data_preprocessing.py, add:
def generate_virtual_samples(df, n_virtual=100):
    """Generate virtual samples using Kernel Density Estimation."""
    # Extract numeric features
    numeric_features = df[FEATURE_COLS_NUM].values
    
    # Fit KDE
    kde = gaussian_kde(numeric_features.T)
    
    # Sample
    virtual_samples = kde.resample(n_virtual).T
    
    # Create DataFrame
    df_virtual = pd.DataFrame(virtual_samples, columns=FEATURE_COLS_NUM)
    
    # Copy categorical features from nearest neighbor
    # ... (implementation details)
    
    return df_virtual

# Use before train/test split
df_augmented = pd.concat([df_original, generate_virtual_samples(df_original)], ignore_index=True)
```

This technique improved R² from 0.05 to 0.99 in Herowati et al. (2024), though results vary.

---

## 📚 Theoretical Background

### Why Log Concentration?

Many adsorption isotherms (Langmuir, Freundlich) predict log-linear relationships between concentration and surface coverage. Including `log_conc_mg_L` helps the model learn this fundamental relationship.

### Why Temperature-Concentration Interaction?

The effect of temperature on IE depends on the adsorption mechanism:
- **Physisorption** (weak bonding): IE decreases with temperature
- **Chemisorption** (strong bonding): IE may increase with temperature

The interaction term `temp × conc` lets the model learn which mechanism dominates at different concentrations.

### Group-Based Cross-Validation

Papers often report multiple experiments under similar conditions. Regular CV could:
1. Train on experiment 1 from Paper A
2. Test on experiment 2 from Paper A

This creates **data leakage** (artificially high performance).

GroupKFold ensures all experiments from a paper are in either train OR test, never both.

### Ensemble Methods

Combining multiple models (Gradient Boosting + Random Forest) provides:
- **Better robustness** - Less sensitive to outliers
- **Lower variance** - More stable predictions
- **Uncertainty estimates** - Variance across models indicates confidence

---

## 🎯 Best Practices & Tips

### Data Quality

✅ **DO:**
- Use standardized experimental methods (Weight Loss, EIS, PDP)
- Include full concentration ranges (0-3000 mg/L)
- Report all experimental conditions
- Use multiple concentrations per inhibitor

❌ **DON'T:**
- Mix different acids in one model (train separate models)
- Include unreliable data (check for outliers!)
- Forget to document data sources
- Use very small datasets (<50 samples) without augmentation

### Model Selection

Based on literature (Akrom 2023, Ma 2023, Haruna 2024):

1. **Gradient Boosting** - Best overall performer (R² > 0.90)
2. **Random Forest** - Good baseline, fast
3. **Ensemble** - Most robust for production use

### Prediction Guidelines

✅ **Reliable predictions when:**
- Concentration: 0-3000 mg/L (within training range)
- Temperature: 25-70°C (common experimental range)
- Inhibitor family is in training data

⚠️ **Use caution when:**
- Concentration > 3000 mg/L (extrapolation)
- Temperature outside training range
- New inhibitor chemical family

❌ **Don't trust predictions for:**
- Completely different acids (HCl, HNO3, etc.)
- Different metals (aluminum, copper, etc.)
- Extreme conditions (T > 100°C, etc.)

---

## 📈 Improving Model Performance

### 1. Add More Data

**Priority additions** (from literature review):
- Lychee peel extract (95.7% @ 3000 mg/L, 0.5M H2SO4)
- *Sida cordifolia* extract (99.0% in H2SO4)
- *Malvaviscus arboreus* extract (97.5% @ 500 mg/L)

See `corrosion_inhibitor_research_sources.md` for 65 H2SO4 studies to mine.

### 2. Add Molecular Descriptors (Advanced)

If you can obtain SMILES strings for inhibitor molecules:

```python
from rdkit import Chem
from rdkit.Chem import Descriptors

def add_molecular_descriptors(df):
    """Add quantum chemical descriptors."""
    df['mol_weight'] = df['smiles'].apply(lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s)))
    df['logP'] = df['smiles'].apply(lambda s: Descriptors.MolLogP(Chem.MolFromSmiles(s)))
    # ... more descriptors
    return df
```

This significantly improved performance in multiple studies (Akrom 2023, El-Idrissi 2023).

### 3. Tune Hyperparameters

Use GridSearchCV for systematic tuning:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(),
    param_grid,
    cv=GroupKFold(5),
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train, groups=groups_train)
best_model = grid_search.best_estimator_
```

### 4. Address Saturation (IE > 95%)

For very high IE values, the model struggles due to saturation. Solutions:

**Option A:** Use `ln_Kads` as target (more linear relationship)

**Option B:** Apply logit transformation:
```python
df['logit_IE'] = np.log((df['IE'] + 0.01) / (100.01 - df['IE']))
# Train on logit_IE, then inverse transform predictions
```

---

## 🐛 Troubleshooting

### Error: "Missing required feature"

**Solution:** Ensure your input data has all required columns. Check column names (case-sensitive!).

### Error: "No module named sklearn"

**Solution:** Install dependencies: `pip install -r requirements.txt`

### Low R² score (< 0.70)

**Possible causes:**
1. Insufficient data (< 100 samples) → Add more data or use virtual samples
2. High noise in measurements → Filter outliers in preprocessing
3. Missing important features → Add molecular descriptors
4. Wrong hyperparameters → Try tuning

### Predictions seem random

**Possible causes:**
1. Model not trained properly → Check TRAINING_REPORT.txt
2. Input data format mismatch → Verify column names and types
3. Extreme extrapolation → Check if inputs are within training range

### Model takes too long to train

**Solutions:**
1. Use HistGradientBoosting instead of GradientBoosting (10x faster)
2. Reduce `n_estimators` (fewer trees)
3. Use fewer CV folds
4. Use `n_jobs=-1` for parallel processing

---

## 📖 References

### Key Papers Implemented

1. **Akrom et al. (2023)** - Gradient Boosting for natural inhibitors
   - Arabian J. Sci. Eng. (2025)
   - DOI: 10.1007/s13369-025-10386-5

2. **Ma et al. (2023)** - Concentration-dependent prediction
   - Corrosion Science
   - DOI: 10.1016/j.corsci.2023.111420

3. **Haruna et al. (2024)** - Ensemble averaging methods
   - Referenced in Akrom et al. (2025)

4. **Chemoinformatics Review (2024)** - IE% saturation issues
   - Molecular Informatics
   - DOI: 10.1002/minf.202400082

### Additional Resources

- Full literature review: `corrosion_inhibitor_research_sources.md`
- 65 H2SO4 plant extract studies compiled
- Adsorption isotherm theory papers
- ML best practices documentation

---

## 💡 Tips for Publication

### Reporting Results

Include in your paper:
1. **Model performance**: R², MAE, RMSE on test set
2. **Cross-validation**: Mean ± std of CV scores
3. **Feature importance**: Top 5-10 features
4. **Uncertainty**: Always report ± error bars
5. **Validation**: Compare predictions to known literature values

### Figures to Include

1. **Predictions vs Actual** (shows model quality)
2. **Feature Importance** (shows what matters)
3. **Concentration-Response Curves** (shows practical utility)
4. **Residuals Plot** (shows systematic errors)

### Comparison to Literature

Benchmark your model against:
- Akrom et al. (2023): R² > 0.90 for natural products
- Ma et al. (2023): 1,241 samples, cross-category prediction
- Your model should achieve similar or better performance

---

## 🤝 Contributing

Want to improve the system? Consider:
- Adding more literature data
- Implementing virtual sample generation
- Adding molecular descriptor calculations
- Creating web interface for predictions
- Extending to other acids (HCl, HNO3)

---

## 📝 License & Citation

If you use this system in your research, please cite:

```
[Your Name], [Year]. "Machine Learning Prediction of Corrosion Inhibition 
Efficiency for Green Plant-Based Inhibitors in H2SO4 Environment"
Based on methods from Akrom et al. (2023) and Ma et al. (2023).
```

---

## 🆘 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the TRAINING_REPORT.txt and PREPROCESSING_REPORT.txt
3. Verify your data format matches the requirements
4. Ensure all dependencies are installed

---

**Last Updated:** January 2026  
**System Version:** 1.0  
**Based on:** 2023-2025 corrosion inhibitor ML literature
