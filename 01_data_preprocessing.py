#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Data Preprocessing for Corrosion Inhibitor ML
======================================================

Based on literature best practices:
- Feature engineering (log concentration, interactions)
- Data quality checks
- Outlier detection
- Train/test splitting with grouped validation

References:
- Akrom et al. (2023) - Gradient Boosting for natural inhibitors
- Ma et al. (2023) - Concentration as feature importance
- Chemoinformatics review (2024) - IE% saturation issues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from scipy import stats

# Configuration
RANDOM_STATE = 42
OUTPUT_DIR = Path("preprocessed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("ENHANCED DATA PREPROCESSING FOR CORROSION INHIBITOR ML")
print("="*70)


def load_raw_data(filepath="corrosion_inhibitors_literature_expanded.csv"):
    """Load raw data from CSV."""
    print(f"\n📂 Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def filter_experimental_conditions(df):
    """
    Filter to H2SO4 and mild steel/carbon steel as per your research focus.
    """
    print("\n🔍 Filtering experimental conditions...")
    
    # Filter to H2SO4
    df_filtered = df[df["acid"].str.upper() == "H2SO4"].copy()
    print(f"   ✓ H2SO4 only: {len(df_filtered)} rows")
    
    # Filter to mild steel / carbon steel / ASTM A36
    steel_pattern = "ASTM A36|mild steel|carbon steel|Q235"
    df_filtered = df_filtered[
        df_filtered["steel_grade"].str.contains(
            steel_pattern, case=False, na=False, regex=True
        )
    ].copy()
    print(f"   ✓ Mild/carbon steel only: {len(df_filtered)} rows")
    
    # Keep only rows with IE data
    df_filtered = df_filtered[df_filtered["inhibition_efficiency_pct"].notna()].copy()
    print(f"   ✓ With IE% data: {len(df_filtered)} rows")
    
    return df_filtered


def add_engineered_features(df):
    """
    Add engineered features based on literature recommendations.
    
    Features added:
    1. log_concentration - Many adsorption isotherms are log-linear
    2. temp_conc_interaction - Temperature affects adsorption differently at different concentrations
    3. acid_strength - Normalized acid molarity
    4. is_blank - Binary indicator for blank/control samples
    """
    print("\n🔧 Engineering features...")
    
    df_eng = df.copy()
    
    # 1. Log concentration (handle zeros by adding small epsilon)
    epsilon = 1e-3  # 0.001 mg/L
    df_eng["log_conc_mg_L"] = np.log10(df_eng["inhibitor_conc_mg_L"] + epsilon)
    print("   ✓ Added: log_conc_mg_L")
    
    # 2. Temperature-concentration interaction
    # Fill missing temperatures with median before interaction
    temp_median = df_eng["temperature_C"].median()
    df_eng["temp_filled"] = df_eng["temperature_C"].fillna(temp_median)
    df_eng["temp_conc_interaction"] = (
        df_eng["temp_filled"] * df_eng["inhibitor_conc_mg_L"] / 1000.0  # Scale down
    )
    print("   ✓ Added: temp_conc_interaction")
    
    # 3. Acid strength (normalized molarity)
    # Most common is 0.5M, normalize to this
    df_eng["acid_strength_norm"] = df_eng["acid_molarity_M"] / 0.5
    print("   ✓ Added: acid_strength_norm")
    
    # 4. Is blank indicator
    df_eng["is_blank"] = (
        df_eng["inhibitor_name"].str.contains("Blank|blank", case=False, na=False)
    ).astype(int)
    print("   ✓ Added: is_blank")
    
    # 5. Immersion time bins (short/medium/long)
    df_eng["immersion_time_bin"] = pd.cut(
        df_eng["immersion_time_h"],
        bins=[0, 6, 24, np.inf],
        labels=["short", "medium", "long"],
        include_lowest=True
    )
    print("   ✓ Added: immersion_time_bin")
    
    # 6. Calculate theoretical ln(Kads) where possible
    # Based on IE% and concentration: θ = IE/100, then Kads = θ/(C*(1-θ))
    # Only for non-saturated IE (< 95%)
    df_eng["surface_coverage"] = df_eng["inhibition_efficiency_pct"] / 100.0
    
    mask = (df_eng["inhibition_efficiency_pct"] < 95) & (df_eng["inhibitor_conc_mg_L"] > 0)
    C_molar = df_eng.loc[mask, "inhibitor_conc_mg_L"] / 1e6  # Very rough approximation
    theta = df_eng.loc[mask, "surface_coverage"]
    
    Kads = theta / (C_molar * (1 - theta + 1e-6))  # Avoid division by zero
    df_eng.loc[mask, "ln_Kads"] = np.log(Kads)
    print("   ✓ Added: ln_Kads (for IE < 95%)")
    
    print(f"\n   Total features now: {len(df_eng.columns)}")
    return df_eng


def check_data_quality(df):
    """
    Perform data quality checks and report issues.
    """
    print("\n📊 Data Quality Report:")
    print("-" * 70)
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        "Missing": missing,
        "Percent": missing_pct
    })
    missing_df = missing_df[missing_df["Missing"] > 0].sort_values("Missing", ascending=False)
    
    if len(missing_df) > 0:
        print("\n⚠️  Missing Values:")
        print(missing_df.to_string())
    else:
        print("\n✓ No missing values!")
    
    # IE% range check
    ie_min, ie_max = df["inhibition_efficiency_pct"].min(), df["inhibition_efficiency_pct"].max()
    print(f"\n📈 IE% range: {ie_min:.2f}% to {ie_max:.2f}%")
    
    if ie_min < 0 or ie_max > 100:
        print("   ⚠️  WARNING: IE% values outside [0, 100] range!")
    
    # Check for duplicate experiments
    duplicates = df.duplicated(
        subset=["inhibitor_name", "acid_molarity_M", "temperature_C", 
                "inhibitor_conc_mg_L", "immersion_time_h"],
        keep=False
    ).sum()
    print(f"\n🔄 Potential duplicate experiments: {duplicates}")
    
    # Distribution of inhibitors
    print(f"\n🧪 Number of unique inhibitors: {df['inhibitor_name'].nunique()}")
    print(f"   Papers: {df['paper_id'].nunique()}")
    
    # Concentration distribution
    print(f"\n💧 Concentration range: {df['inhibitor_conc_mg_L'].min():.0f} to {df['inhibitor_conc_mg_L'].max():.0f} mg/L")
    
    return missing_df


def detect_outliers(df, column="inhibition_efficiency_pct"):
    """
    Detect potential outliers using IQR method.
    """
    print(f"\n🎯 Outlier Detection for '{column}':")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"   IQR range: [{Q1:.2f}, {Q3:.2f}]")
    print(f"   Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"   Number of outliers: {len(outliers)} ({100*len(outliers)/len(df):.1f}%)")
    
    return outliers


def create_train_test_split(df, test_size=0.2, val_size=0.1):
    """
    Create train/validation/test splits with group-based splitting.
    
    Groups by paper_id to avoid data leakage.
    """
    print(f"\n📊 Creating train/val/test splits...")
    print(f"   Test size: {test_size*100:.0f}%")
    print(f"   Validation size: {val_size*100:.0f}%")
    
    # First split: train+val vs test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df["paper_id"]))
    
    df_train_val = df.iloc[train_val_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val["paper_id"]))
    
    df_train = df_train_val.iloc[train_idx].copy()
    df_val = df_train_val.iloc[val_idx].copy()
    
    print(f"\n   ✓ Train set: {len(df_train)} samples ({len(df_train['paper_id'].unique())} papers)")
    print(f"   ✓ Val set:   {len(df_val)} samples ({len(df_val['paper_id'].unique())} papers)")
    print(f"   ✓ Test set:  {len(df_test)} samples ({len(df_test['paper_id'].unique())} papers)")
    
    # Check for group leakage
    train_papers = set(df_train["paper_id"].unique())
    val_papers = set(df_val["paper_id"].unique())
    test_papers = set(df_test["paper_id"].unique())
    
    if len(train_papers & test_papers) > 0:
        print("   ⚠️  WARNING: Paper leakage between train and test!")
    if len(train_papers & val_papers) > 0:
        print("   ⚠️  WARNING: Paper leakage between train and val!")
    if len(val_papers & test_papers) > 0:
        print("   ⚠️  WARNING: Paper leakage between val and test!")
    else:
        print("   ✓ No group leakage detected!")
    
    return df_train, df_val, df_test


def visualize_data_distribution(df):
    """
    Create visualization of data distributions.
    """
    print("\n📊 Creating data distribution visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Data Distribution Analysis", fontsize=16, fontweight="bold")
    
    # 1. IE% distribution
    ax = axes[0, 0]
    ax.hist(df["inhibition_efficiency_pct"], bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Inhibition Efficiency (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("IE% Distribution")
    ax.axvline(df["inhibition_efficiency_pct"].median(), color="red", linestyle="--", 
               label=f'Median: {df["inhibition_efficiency_pct"].median():.1f}%')
    ax.legend()
    
    # 2. Concentration distribution (log scale)
    ax = axes[0, 1]
    conc_nonzero = df[df["inhibitor_conc_mg_L"] > 0]["inhibitor_conc_mg_L"]
    ax.hist(np.log10(conc_nonzero), bins=30, edgecolor="black", alpha=0.7, color="green")
    ax.set_xlabel("log10(Concentration [mg/L])")
    ax.set_ylabel("Frequency")
    ax.set_title("Concentration Distribution (log scale)")
    
    # 3. Temperature distribution
    ax = axes[0, 2]
    temp_data = df["temperature_C"].dropna()
    ax.hist(temp_data, bins=20, edgecolor="black", alpha=0.7, color="orange")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Frequency")
    ax.set_title("Temperature Distribution")
    
    # 4. IE% vs Concentration
    ax = axes[1, 0]
    scatter_data = df[df["inhibitor_conc_mg_L"] > 0]
    ax.scatter(scatter_data["inhibitor_conc_mg_L"], 
               scatter_data["inhibition_efficiency_pct"],
               alpha=0.5, s=30)
    ax.set_xlabel("Concentration (mg/L)")
    ax.set_ylabel("IE (%)")
    ax.set_title("IE% vs Concentration")
    ax.set_xscale("log")
    
    # 5. IE% by inhibitor (top 10)
    ax = axes[1, 1]
    top_inhibitors = df["inhibitor_name"].value_counts().head(10).index
    df_top = df[df["inhibitor_name"].isin(top_inhibitors)]
    df_top.boxplot(column="inhibition_efficiency_pct", by="inhibitor_name", ax=ax, rot=45)
    ax.set_xlabel("")
    ax.set_ylabel("IE (%)")
    ax.set_title("IE% Distribution by Top 10 Inhibitors")
    plt.sca(ax)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    
    # 6. Sample count by paper
    ax = axes[1, 2]
    paper_counts = df["paper_id"].value_counts().head(15)
    ax.barh(range(len(paper_counts)), paper_counts.values, color="skyblue", edgecolor="black")
    ax.set_yticks(range(len(paper_counts)))
    ax.set_yticklabels([p[:30] for p in paper_counts.index], fontsize=8)
    ax.set_xlabel("Number of Samples")
    ax.set_title("Samples per Paper (Top 15)")
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "data_distribution_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"   ✓ Saved: {output_path}")


def save_datasets(df_train, df_val, df_test, df_full):
    """
    Save processed datasets to CSV files.
    """
    print("\n💾 Saving processed datasets...")
    
    # Save splits
    df_train.to_csv(OUTPUT_DIR / "train_data.csv", index=False)
    print(f"   ✓ train_data.csv ({len(df_train)} rows)")
    
    df_val.to_csv(OUTPUT_DIR / "val_data.csv", index=False)
    print(f"   ✓ val_data.csv ({len(df_val)} rows)")
    
    df_test.to_csv(OUTPUT_DIR / "test_data.csv", index=False)
    print(f"   ✓ test_data.csv ({len(df_test)} rows)")
    
    # Save full processed dataset
    df_full.to_csv(OUTPUT_DIR / "full_processed_data.csv", index=False)
    print(f"   ✓ full_processed_data.csv ({len(df_full)} rows)")
    
    # Create a feature documentation file
    feature_docs = """
# Feature Documentation

## Original Features
- `paper_id`: Source paper identifier
- `inhibitor_name`: Name of the corrosion inhibitor
- `inhibitor_scientific`: Scientific name of plant/compound
- `steel_grade`: Type of steel tested
- `acid`: Type of acid (H2SO4)
- `acid_molarity_M`: Acid concentration in molarity
- `temperature_C`: Test temperature in Celsius
- `immersion_time_h`: Immersion time in hours
- `inhibitor_conc_mg_L`: Inhibitor concentration in mg/L
- `method`: Experimental method (e.g., Weight Loss, EIS)
- `inhibition_efficiency_pct`: Target variable (0-100%)

## Engineered Features
- `log_conc_mg_L`: Log10 of inhibitor concentration
  - Rationale: Many adsorption isotherms are log-linear
  - Range: log10(0.001) to log10(max_concentration)

- `temp_conc_interaction`: Temperature × Concentration / 1000
  - Rationale: Temperature effects vary with concentration
  - Captures physisorption vs chemisorption behavior

- `acid_strength_norm`: Acid molarity normalized to 0.5M
  - Rationale: Standardizes across different acid concentrations
  - Value: acid_molarity_M / 0.5

- `is_blank`: Binary indicator for control samples
  - 1 if inhibitor_name contains "Blank"
  - 0 otherwise

- `immersion_time_bin`: Categorical time bins
  - "short": 0-6 hours
  - "medium": 6-24 hours
  - "long": >24 hours

- `surface_coverage`: IE% / 100
  - Theoretical surface coverage (θ)

- `ln_Kads`: Natural log of adsorption equilibrium constant
  - Only calculated for IE < 95% (avoid saturation)
  - Based on Langmuir isotherm: Kads = θ/(C*(1-θ))
  - Can be used as alternative target variable

## Feature Usage Recommendations

**For Standard IE% Prediction:**
- Use: All numeric + categorical features
- Target: `inhibition_efficiency_pct`

**For Adsorption Modeling:**
- Use: log_conc_mg_L, temperature_C, temp_conc_interaction
- Target: `ln_Kads` (for samples where available)

**For Concentration-Response Curves:**
- Vary: `inhibitor_conc_mg_L` or `log_conc_mg_L`
- Fix: temperature_C, acid_molarity_M, steel_grade, method

**Temperature Studies:**
- Vary: `temperature_C`
- Include: `temp_conc_interaction` to capture non-linear effects
"""
    
    with open(OUTPUT_DIR / "FEATURE_DOCUMENTATION.txt", "w") as f:
        f.write(feature_docs)
    print(f"   ✓ FEATURE_DOCUMENTATION.txt")


def create_summary_report(df_full, df_train, df_val, df_test):
    """
    Create a summary report of the preprocessing.
    """
    print("\n📄 Creating summary report...")
    
    report = f"""
{'='*70}
DATA PREPROCESSING SUMMARY REPORT
{'='*70}

Dataset: Corrosion Inhibitors in H2SO4 Environment
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
1. DATA OVERVIEW
{'='*70}

Total samples (after filtering): {len(df_full)}
Unique inhibitors: {df_full['inhibitor_name'].nunique()}
Unique papers: {df_full['paper_id'].nunique()}

Experimental Conditions:
- Acid: H2SO4 only
- Steel: Mild/carbon steel, ASTM A36, Q235
- Method: {', '.join(df_full['method'].unique())}

{'='*70}
2. TARGET VARIABLE STATISTICS
{'='*70}

Inhibition Efficiency (%):
  Mean:   {df_full['inhibition_efficiency_pct'].mean():.2f}%
  Median: {df_full['inhibition_efficiency_pct'].median():.2f}%
  Std:    {df_full['inhibition_efficiency_pct'].std():.2f}%
  Min:    {df_full['inhibition_efficiency_pct'].min():.2f}%
  Max:    {df_full['inhibition_efficiency_pct'].max():.2f}%

Distribution:
  IE < 50%:    {(df_full['inhibition_efficiency_pct'] < 50).sum()} samples ({100*(df_full['inhibition_efficiency_pct'] < 50).sum()/len(df_full):.1f}%)
  50% ≤ IE < 80%: {((df_full['inhibition_efficiency_pct'] >= 50) & (df_full['inhibition_efficiency_pct'] < 80)).sum()} samples ({100*((df_full['inhibition_efficiency_pct'] >= 50) & (df_full['inhibition_efficiency_pct'] < 80)).sum()/len(df_full):.1f}%)
  IE ≥ 80%:    {(df_full['inhibition_efficiency_pct'] >= 80).sum()} samples ({100*(df_full['inhibition_efficiency_pct'] >= 80).sum()/len(df_full):.1f}%)

{'='*70}
3. FEATURE RANGES
{'='*70}

Inhibitor Concentration:
  Min:    {df_full['inhibitor_conc_mg_L'].min():.1f} mg/L
  Max:    {df_full['inhibitor_conc_mg_L'].max():.1f} mg/L
  Median: {df_full['inhibitor_conc_mg_L'].median():.1f} mg/L

Temperature:
  Min:    {df_full['temperature_C'].min():.1f}°C
  Max:    {df_full['temperature_C'].max():.1f}°C
  Median: {df_full['temperature_C'].median():.1f}°C

Acid Molarity:
  Min:    {df_full['acid_molarity_M'].min():.2f} M
  Max:    {df_full['acid_molarity_M'].max():.2f} M
  Median: {df_full['acid_molarity_M'].median():.2f} M

Immersion Time:
  Min:    {df_full['immersion_time_h'].min():.1f} hours
  Max:    {df_full['immersion_time_h'].max():.1f} hours
  Median: {df_full['immersion_time_h'].median():.1f} hours

{'='*70}
4. TRAIN/VAL/TEST SPLIT
{'='*70}

Training Set:
  Samples: {len(df_train)} ({100*len(df_train)/len(df_full):.1f}%)
  Papers:  {df_train['paper_id'].nunique()}
  Inhibitors: {df_train['inhibitor_name'].nunique()}

Validation Set:
  Samples: {len(df_val)} ({100*len(df_val)/len(df_full):.1f}%)
  Papers:  {df_val['paper_id'].nunique()}
  Inhibitors: {df_val['inhibitor_name'].nunique()}

Test Set:
  Samples: {len(df_test)} ({100*len(df_test)/len(df_full):.1f}%)
  Papers:  {df_test['paper_id'].nunique()}
  Inhibitors: {df_test['inhibitor_name'].nunique()}

Split Strategy: GroupShuffleSplit by paper_id (prevents data leakage)

{'='*70}
5. TOP PERFORMING INHIBITORS
{'='*70}

Top 10 by Mean IE%:
"""
    
    # Add top inhibitors
    top_inhibitors = df_full.groupby("inhibitor_name")["inhibition_efficiency_pct"].agg(
        ["mean", "std", "count"]
    ).sort_values("mean", ascending=False).head(10)
    
    for idx, (inhibitor, row) in enumerate(top_inhibitors.iterrows(), 1):
        report += f"\n{idx:2d}. {inhibitor[:50]:<50s}  {row['mean']:5.1f}% ± {row['std']:4.1f}% (n={int(row['count'])})"
    
    report += f"""

{'='*70}
6. ENGINEERED FEATURES ADDED
{'='*70}

✓ log_conc_mg_L          - Log-transformed concentration
✓ temp_conc_interaction  - Temperature × Concentration interaction
✓ acid_strength_norm     - Normalized acid molarity
✓ is_blank               - Binary blank indicator
✓ immersion_time_bin     - Categorical time bins
✓ ln_Kads                - Adsorption constant (where IE < 95%)

See FEATURE_DOCUMENTATION.txt for detailed descriptions.

{'='*70}
7. DATA QUALITY NOTES
{'='*70}

✓ Filtered to H2SO4 environment only
✓ Filtered to mild/carbon steel only
✓ Removed rows without IE% data
✓ Added robust feature engineering
✓ Group-based train/val/test split (no paper leakage)

{'='*70}
8. NEXT STEPS
{'='*70}

1. Review the data distribution visualization (data_distribution_analysis.png)
2. Run the ML training script (02_ml_training_enhanced.py)
3. Consider virtual sample generation if needed (for small dataset augmentation)
4. Experiment with both IE% and ln_Kads as target variables

{'='*70}
END OF REPORT
{'='*70}
"""
    
    # Save report
    with open(OUTPUT_DIR / "PREPROCESSING_REPORT.txt", "w") as f:
        f.write(report)
    
    print(f"   ✓ PREPROCESSING_REPORT.txt")
    
    # Print to console as well
    print("\n" + report)


def main():
    """Main preprocessing pipeline."""
    
    # 1. Load data
    df = load_raw_data()
    
    # 2. Filter to experimental conditions
    df_filtered = filter_experimental_conditions(df)
    
    # 3. Add engineered features
    df_engineered = add_engineered_features(df_filtered)
    
    # 4. Data quality checks
    missing_report = check_data_quality(df_engineered)
    
    # 5. Outlier detection
    outliers = detect_outliers(df_engineered)
    
    # 6. Create train/val/test splits
    df_train, df_val, df_test = create_train_test_split(df_engineered)
    
    # 7. Visualize distributions
    visualize_data_distribution(df_engineered)
    
    # 8. Save all datasets
    save_datasets(df_train, df_val, df_test, df_engineered)
    
    # 9. Create summary report
    create_summary_report(df_engineered, df_train, df_val, df_test)
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print("\nFiles created:")
    print("  • train_data.csv - Training dataset")
    print("  • val_data.csv - Validation dataset")
    print("  • test_data.csv - Test dataset")
    print("  • full_processed_data.csv - Complete processed dataset")
    print("  • data_distribution_analysis.png - Data visualizations")
    print("  • FEATURE_DOCUMENTATION.txt - Feature descriptions")
    print("  • PREPROCESSING_REPORT.txt - Detailed summary report")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
