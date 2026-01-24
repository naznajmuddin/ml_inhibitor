#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Graph Generation for Corrosion Inhibitor Study
=============================================================

Generates publication-quality figures for the 3 main inhibitors:
- Curry leaf extract (Murraya koenigii)
- Peanut shell extract (Arachis hypogaea)
- Aloe vera extract

Usage:
    python 06_graphs.py              # Generate all graphs
    python 06_graphs.py --category concentration  # Specific category
    python 06_graphs.py --list       # List available graph categories
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
import seaborn as sns
import joblib
import argparse
from scipy import stats
from scipy.optimize import curve_fit

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme for inhibitors
INHIBITOR_COLORS = {
    'Curry leaf extract': '#2ecc71',      # Green
    'Peanut shell extract': '#e74c3c',    # Red
    'Aloe vera extract': '#3498db',       # Blue
}

INHIBITOR_MARKERS = {
    'Curry leaf extract': 'o',
    'Peanut shell extract': 's',
    'Aloe vera extract': '^',
}

INHIBITOR_SCIENTIFIC = {
    'Curry leaf extract': 'Murraya koenigii',
    'Peanut shell extract': 'Arachis hypogaea',
    'Aloe vera extract': 'Aloe vera',
}

# Acid colors
ACID_COLORS = {'H2SO4': '#e74c3c', 'HCl': '#3498db'}

OUTPUT_DIR = Path("study_figures")


def load_data():
    """Load all required data."""
    print("\n[INFO] Loading data...")

    # Load main dataset
    df = pd.read_csv("corrosion_inhibitors_expanded_v2.csv")
    print(f"   Main dataset: {len(df)} rows")

    # Load preprocessed data if available
    train_path = Path("preprocessed_data/train_data.csv")
    if train_path.exists():
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv("preprocessed_data/val_data.csv")
        df_test = pd.read_csv("preprocessed_data/test_data.csv")
        print(f"   Train/Val/Test: {len(df_train)}/{len(df_val)}/{len(df_test)}")
    else:
        df_train = df_val = df_test = None

    # Load experimental electrochemistry data
    echem_path = Path("ElectrochemistryResults.csv")
    if echem_path.exists():
        df_echem = pd.read_csv(echem_path)
        print(f"   Electrochemistry data: {len(df_echem)} rows")
    else:
        df_echem = None

    return df, df_train, df_val, df_test, df_echem


def load_models():
    """Load trained ML models."""
    print("\n[INFO] Loading ML models...")

    models = {}
    model_dir = Path("ml_models")

    # Load preprocessor
    prep_path = model_dir / "best_model_preprocessor.pkl"
    if prep_path.exists():
        models['preprocessor'] = joblib.load(prep_path)
        print("   Preprocessor loaded")

    # Load IE model
    ie_path = model_dir / "best_model_IE_model.pkl"
    if ie_path.exists():
        models['IE'] = joblib.load(ie_path)
        print("   IE model loaded")

    # Load Ecorr model
    ecorr_path = model_dir / "best_model_Ecorr_model.pkl"
    if ecorr_path.exists():
        models['Ecorr'] = joblib.load(ecorr_path)
        print("   Ecorr model loaded")

    # Load Icorr model
    icorr_path = model_dir / "best_model_Icorr_model.pkl"
    if icorr_path.exists():
        models['Icorr'] = joblib.load(icorr_path)
        print("   Icorr model loaded")

    return models


# =============================================================================
# 1. CONCENTRATION-RESPONSE GRAPHS
# =============================================================================

def plot_concentration_vs_ie(df):
    """Plot IE% vs concentration for each inhibitor."""
    print("\n[GRAPH] Concentration vs IE% curves...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (inhibitor, color) in enumerate(INHIBITOR_COLORS.items()):
        ax = axes[idx]
        data = df[df['inhibitor_name'] == inhibitor]

        if len(data) == 0:
            continue

        # Group by concentration and calculate mean/std
        grouped = data.groupby('inhibitor_conc_mg_L')['inhibition_efficiency_pct'].agg(['mean', 'std', 'count'])
        grouped = grouped.reset_index()

        # Plot with error bars
        ax.errorbar(grouped['inhibitor_conc_mg_L'], grouped['mean'],
                   yerr=grouped['std'].fillna(0),
                   fmt=f'{INHIBITOR_MARKERS[inhibitor]}-', color=color,
                   capsize=4, capthick=1.5, linewidth=2, markersize=8,
                   label=f'{inhibitor}\n({INHIBITOR_SCIENTIFIC[inhibitor]})')

        ax.set_xlabel('Concentration (mg/L)')
        ax.set_ylabel('Inhibition Efficiency (%)')
        ax.set_title(inhibitor)
        ax.set_ylim(-10, 105)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize=9)

    plt.suptitle('Concentration-Response Curves for Green Corrosion Inhibitors',
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_concentration_vs_ie_individual.png")
    plt.close()
    print(f"   Saved: 01_concentration_vs_ie_individual.png")


def plot_concentration_comparison(df):
    """Compare all inhibitors on single plot."""
    print("\n[GRAPH] Concentration comparison (all inhibitors)...")

    fig, ax = plt.subplots(figsize=(10, 7))

    for inhibitor, color in INHIBITOR_COLORS.items():
        data = df[df['inhibitor_name'] == inhibitor]

        if len(data) == 0:
            continue

        # Group by concentration
        grouped = data.groupby('inhibitor_conc_mg_L')['inhibition_efficiency_pct'].agg(['mean', 'std'])
        grouped = grouped.reset_index()

        ax.errorbar(grouped['inhibitor_conc_mg_L'], grouped['mean'],
                   yerr=grouped['std'].fillna(0),
                   fmt=f'{INHIBITOR_MARKERS[inhibitor]}-', color=color,
                   capsize=3, linewidth=2, markersize=8,
                   label=f'{inhibitor} ({INHIBITOR_SCIENTIFIC[inhibitor]})')

    ax.set_xlabel('Inhibitor Concentration (mg/L)', fontweight='bold')
    ax.set_ylabel('Inhibition Efficiency (%)', fontweight='bold')
    ax.set_title('Comparison of Green Corrosion Inhibitors\nin Acidic Medium on Mild Steel',
                fontweight='bold')
    ax.set_ylim(-10, 105)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_concentration_comparison_all.png")
    plt.close()
    print(f"   Saved: 02_concentration_comparison_all.png")


def plot_concentration_log_scale(df):
    """Plot concentration on log scale."""
    print("\n[GRAPH] Concentration (log scale) vs IE%...")

    fig, ax = plt.subplots(figsize=(10, 7))

    for inhibitor, color in INHIBITOR_COLORS.items():
        data = df[(df['inhibitor_name'] == inhibitor) & (df['inhibitor_conc_mg_L'] > 0)]

        if len(data) == 0:
            continue

        grouped = data.groupby('inhibitor_conc_mg_L')['inhibition_efficiency_pct'].agg(['mean', 'std'])
        grouped = grouped.reset_index()

        ax.semilogx(grouped['inhibitor_conc_mg_L'], grouped['mean'],
                   f'{INHIBITOR_MARKERS[inhibitor]}-', color=color,
                   linewidth=2, markersize=8, label=inhibitor)

        # Add shaded uncertainty region
        ax.fill_between(grouped['inhibitor_conc_mg_L'],
                       grouped['mean'] - grouped['std'].fillna(0),
                       grouped['mean'] + grouped['std'].fillna(0),
                       alpha=0.2, color=color)

    ax.set_xlabel('Inhibitor Concentration (mg/L) - Log Scale', fontweight='bold')
    ax.set_ylabel('Inhibition Efficiency (%)', fontweight='bold')
    ax.set_title('Concentration-Response Curves (Logarithmic Scale)', fontweight='bold')
    ax.set_ylim(-10, 105)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_concentration_log_scale.png")
    plt.close()
    print(f"   Saved: 03_concentration_log_scale.png")


# =============================================================================
# 2. ACID TYPE COMPARISON
# =============================================================================

def plot_acid_comparison(df):
    """Compare inhibitor performance in different acids."""
    print("\n[GRAPH] Acid type comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart comparison
    ax1 = axes[0]

    # Get data for each inhibitor and acid
    inhibitors = list(INHIBITOR_COLORS.keys())
    acids = ['H2SO4', 'HCl']
    x = np.arange(len(inhibitors))
    width = 0.35

    for i, acid in enumerate(acids):
        means = []
        stds = []
        for inhibitor in inhibitors:
            data = df[(df['inhibitor_name'] == inhibitor) &
                      (df['acid'] == acid) &
                      (df['inhibitor_conc_mg_L'] > 0)]
            if len(data) > 0:
                means.append(data['inhibition_efficiency_pct'].mean())
                stds.append(data['inhibition_efficiency_pct'].std())
            else:
                means.append(0)
                stds.append(0)

        offset = width * (i - 0.5)
        bars = ax1.bar(x + offset, means, width, yerr=stds,
                      label=acid, color=ACID_COLORS[acid],
                      capsize=4, alpha=0.8)

    ax1.set_xlabel('Inhibitor', fontweight='bold')
    ax1.set_ylabel('Mean Inhibition Efficiency (%)', fontweight='bold')
    ax1.set_title('Performance Comparison: H₂SO₄ vs HCl', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([inh.replace(' extract', '') for inh in inhibitors], rotation=15)
    ax1.legend(title='Acid Medium')
    ax1.set_ylim(0, 100)

    # Grouped scatter plot
    ax2 = axes[1]

    for inhibitor, color in INHIBITOR_COLORS.items():
        for acid, marker in [('H2SO4', 'o'), ('HCl', 's')]:
            data = df[(df['inhibitor_name'] == inhibitor) &
                      (df['acid'] == acid) &
                      (df['inhibitor_conc_mg_L'] > 0)]
            if len(data) > 0:
                ax2.scatter(data['inhibitor_conc_mg_L'],
                           data['inhibition_efficiency_pct'],
                           c=color, marker=marker, s=60, alpha=0.7,
                           label=f'{inhibitor.split()[0]} - {acid}')

    ax2.set_xlabel('Concentration (mg/L)', fontweight='bold')
    ax2.set_ylabel('Inhibition Efficiency (%)', fontweight='bold')
    ax2.set_title('IE% Distribution by Acid Type', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8, ncol=2)
    ax2.set_ylim(-10, 105)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_acid_comparison.png")
    plt.close()
    print(f"   Saved: 04_acid_comparison.png")


# =============================================================================
# 3. TEMPERATURE EFFECT
# =============================================================================

def plot_temperature_effect(df):
    """Plot effect of temperature on IE%."""
    print("\n[GRAPH] Temperature effect...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Temperature vs IE% for each inhibitor
    ax1 = axes[0]

    for inhibitor, color in INHIBITOR_COLORS.items():
        data = df[(df['inhibitor_name'] == inhibitor) & (df['inhibitor_conc_mg_L'] > 0)]

        if len(data) == 0:
            continue

        grouped = data.groupby('temperature_C')['inhibition_efficiency_pct'].agg(['mean', 'std'])
        grouped = grouped.reset_index()

        if len(grouped) > 1:
            ax1.errorbar(grouped['temperature_C'], grouped['mean'],
                        yerr=grouped['std'].fillna(0),
                        fmt=f'{INHIBITOR_MARKERS[inhibitor]}-', color=color,
                        capsize=4, linewidth=2, markersize=8, label=inhibitor)

    ax1.set_xlabel('Temperature (°C)', fontweight='bold')
    ax1.set_ylabel('Mean Inhibition Efficiency (%)', fontweight='bold')
    ax1.set_title('Effect of Temperature on Inhibition Efficiency', fontweight='bold')
    ax1.legend(loc='best')
    ax1.set_ylim(0, 100)

    # Heatmap of temperature vs concentration
    ax2 = axes[1]

    # Create pivot table for heatmap (use Aloe vera as example - has most temperature data)
    aloe_data = df[(df['inhibitor_name'] == 'Aloe vera extract') &
                   (df['inhibitor_conc_mg_L'] > 0)]

    if len(aloe_data) > 0:
        pivot = aloe_data.pivot_table(
            values='inhibition_efficiency_pct',
            index='temperature_C',
            columns='inhibitor_conc_mg_L',
            aggfunc='mean'
        )

        if pivot.size > 0:
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                       ax=ax2, cbar_kws={'label': 'IE (%)'})
            ax2.set_xlabel('Concentration (mg/L)', fontweight='bold')
            ax2.set_ylabel('Temperature (°C)', fontweight='bold')
            ax2.set_title('Aloe vera: Temperature × Concentration\nHeatmap', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_temperature_effect.png")
    plt.close()
    print(f"   Saved: 05_temperature_effect.png")


# =============================================================================
# 4. ADSORPTION ISOTHERM
# =============================================================================

def langmuir_isotherm(C, Kads):
    """Langmuir isotherm: theta = KC / (1 + KC)"""
    return Kads * C / (1 + Kads * C)


def plot_adsorption_isotherm(df):
    """Plot Langmuir adsorption isotherm."""
    print("\n[GRAPH] Langmuir adsorption isotherm...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # C/theta vs C (linearized Langmuir)
    ax1 = axes[0]

    for inhibitor, color in INHIBITOR_COLORS.items():
        data = df[(df['inhibitor_name'] == inhibitor) &
                  (df['inhibitor_conc_mg_L'] > 0) &
                  (df['inhibition_efficiency_pct'] > 0) &
                  (df['inhibition_efficiency_pct'] < 100)]

        if len(data) < 3:
            continue

        # Calculate surface coverage (theta = IE/100)
        C = data['inhibitor_conc_mg_L'].values
        theta = data['inhibition_efficiency_pct'].values / 100

        # Langmuir linearization: C/theta vs C
        C_over_theta = C / theta

        # Linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(C, C_over_theta)

        ax1.scatter(C, C_over_theta, c=color, marker=INHIBITOR_MARKERS[inhibitor],
                   s=60, label=f'{inhibitor} (R²={r_value**2:.4f})')

        # Fit line
        C_fit = np.linspace(C.min(), C.max(), 100)
        ax1.plot(C_fit, slope * C_fit + intercept, '--', color=color, alpha=0.7)

    ax1.set_xlabel('Concentration, C (mg/L)', fontweight='bold')
    ax1.set_ylabel('C/θ (mg/L)', fontweight='bold')
    ax1.set_title('Langmuir Adsorption Isotherm\n(Linearized Form)', fontweight='bold')
    ax1.legend(loc='best', fontsize=9)

    # Adsorption curve (theta vs C)
    ax2 = axes[1]

    for inhibitor, color in INHIBITOR_COLORS.items():
        data = df[(df['inhibitor_name'] == inhibitor) &
                  (df['inhibitor_conc_mg_L'] > 0) &
                  (df['inhibition_efficiency_pct'] > 0)]

        if len(data) < 3:
            continue

        grouped = data.groupby('inhibitor_conc_mg_L')['inhibition_efficiency_pct'].mean()
        C = grouped.index.values
        theta = grouped.values / 100

        ax2.plot(C, theta, f'{INHIBITOR_MARKERS[inhibitor]}-', color=color,
                linewidth=2, markersize=8, label=inhibitor)

    ax2.set_xlabel('Concentration (mg/L)', fontweight='bold')
    ax2.set_ylabel('Surface Coverage (θ)', fontweight='bold')
    ax2.set_title('Surface Coverage vs Concentration', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_adsorption_isotherm.png")
    plt.close()
    print(f"   Saved: 06_adsorption_isotherm.png")


# =============================================================================
# 5. ELECTROCHEMICAL DATA
# =============================================================================

def plot_electrochemical_data(df, df_echem):
    """Plot electrochemical parameters (Ecorr, Icorr)."""
    print("\n[GRAPH] Electrochemical parameters...")

    # Filter data with electrochemical measurements
    echem_data = df[df['Ecorr_mV'].notna()]

    if len(echem_data) < 3:
        print("   [SKIP] Insufficient electrochemical data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Ecorr vs Concentration
    ax1 = axes[0, 0]
    for inhibitor, color in INHIBITOR_COLORS.items():
        data = echem_data[echem_data['inhibitor_name'] == inhibitor]
        if len(data) > 0:
            ax1.scatter(data['inhibitor_conc_mg_L'], data['Ecorr_mV'],
                       c=color, marker=INHIBITOR_MARKERS[inhibitor],
                       s=80, label=inhibitor)

    ax1.set_xlabel('Concentration (mg/L)', fontweight='bold')
    ax1.set_ylabel('Ecorr (mV vs SCE)', fontweight='bold')
    ax1.set_title('Corrosion Potential vs Concentration', fontweight='bold')
    ax1.legend(loc='best')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 2. Icorr vs Concentration (log scale)
    ax2 = axes[0, 1]
    for inhibitor, color in INHIBITOR_COLORS.items():
        data = echem_data[(echem_data['inhibitor_name'] == inhibitor) &
                          (echem_data['Icorr_uA_cm2'] > 0)]
        if len(data) > 0:
            ax2.semilogy(data['inhibitor_conc_mg_L'], data['Icorr_uA_cm2'],
                        f'{INHIBITOR_MARKERS[inhibitor]}-', color=color,
                        markersize=8, linewidth=2, label=inhibitor)

    ax2.set_xlabel('Concentration (mg/L)', fontweight='bold')
    ax2.set_ylabel('Icorr (μA/cm²) - Log Scale', fontweight='bold')
    ax2.set_title('Corrosion Current Density vs Concentration', fontweight='bold')
    ax2.legend(loc='best')

    # 3. Ecorr vs IE%
    ax3 = axes[1, 0]
    for inhibitor, color in INHIBITOR_COLORS.items():
        data = echem_data[echem_data['inhibitor_name'] == inhibitor]
        if len(data) > 0:
            ax3.scatter(data['Ecorr_mV'], data['inhibition_efficiency_pct'],
                       c=color, marker=INHIBITOR_MARKERS[inhibitor],
                       s=80, label=inhibitor)

    ax3.set_xlabel('Ecorr (mV vs SCE)', fontweight='bold')
    ax3.set_ylabel('Inhibition Efficiency (%)', fontweight='bold')
    ax3.set_title('IE% vs Corrosion Potential', fontweight='bold')
    ax3.legend(loc='best')

    # 4. Icorr vs IE%
    ax4 = axes[1, 1]
    for inhibitor, color in INHIBITOR_COLORS.items():
        data = echem_data[(echem_data['inhibitor_name'] == inhibitor) &
                          (echem_data['Icorr_uA_cm2'] > 0)]
        if len(data) > 0:
            ax4.scatter(data['Icorr_uA_cm2'], data['inhibition_efficiency_pct'],
                       c=color, marker=INHIBITOR_MARKERS[inhibitor],
                       s=80, label=inhibitor)

    ax4.set_xlabel('Icorr (μA/cm²)', fontweight='bold')
    ax4.set_ylabel('Inhibition Efficiency (%)', fontweight='bold')
    ax4.set_title('IE% vs Corrosion Current Density', fontweight='bold')
    ax4.legend(loc='best')
    ax4.set_xscale('log')

    plt.suptitle('Electrochemical Parameters Analysis', fontweight='bold', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_electrochemical_data.png")
    plt.close()
    print(f"   Saved: 07_electrochemical_data.png")


# =============================================================================
# 6. BAR CHARTS & COMPARISONS
# =============================================================================

def plot_max_ie_comparison(df):
    """Compare maximum IE% achieved by each inhibitor."""
    print("\n[GRAPH] Maximum IE% comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart of max IE%
    ax1 = axes[0]

    inhibitors = list(INHIBITOR_COLORS.keys())
    max_ie = []
    optimal_conc = []

    for inhibitor in inhibitors:
        data = df[df['inhibitor_name'] == inhibitor]
        if len(data) > 0:
            max_idx = data['inhibition_efficiency_pct'].idxmax()
            max_ie.append(data.loc[max_idx, 'inhibition_efficiency_pct'])
            optimal_conc.append(data.loc[max_idx, 'inhibitor_conc_mg_L'])
        else:
            max_ie.append(0)
            optimal_conc.append(0)

    colors = [INHIBITOR_COLORS[inh] for inh in inhibitors]
    bars = ax1.bar(range(len(inhibitors)), max_ie, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val, conc in zip(bars, max_ie, optimal_conc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%\n({conc:.0f} mg/L)', ha='center', va='bottom', fontsize=10)

    ax1.set_xticks(range(len(inhibitors)))
    ax1.set_xticklabels([inh.replace(' extract', '') for inh in inhibitors])
    ax1.set_ylabel('Maximum IE (%)', fontweight='bold')
    ax1.set_title('Maximum Inhibition Efficiency Achieved', fontweight='bold')
    ax1.set_ylim(0, 110)

    # Optimal concentration comparison
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(inhibitors)), optimal_conc, color=colors,
                    edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars2, optimal_conc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_xticks(range(len(inhibitors)))
    ax2.set_xticklabels([inh.replace(' extract', '') for inh in inhibitors])
    ax2.set_ylabel('Optimal Concentration (mg/L)', fontweight='bold')
    ax2.set_title('Optimal Concentration for Maximum IE%', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_max_ie_comparison.png")
    plt.close()
    print(f"   Saved: 08_max_ie_comparison.png")


def plot_ie_distribution(df):
    """Box plots showing IE% distribution."""
    print("\n[GRAPH] IE% distribution (box plots)...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot by inhibitor
    ax1 = axes[0]

    inhibitors = list(INHIBITOR_COLORS.keys())
    data_list = []
    for inhibitor in inhibitors:
        data = df[(df['inhibitor_name'] == inhibitor) & (df['inhibitor_conc_mg_L'] > 0)]
        data_list.append(data['inhibition_efficiency_pct'].values)

    bp = ax1.boxplot(data_list, patch_artist=True, tick_labels=[inh.replace(' extract', '') for inh in inhibitors])

    for patch, inhibitor in zip(bp['boxes'], inhibitors):
        patch.set_facecolor(INHIBITOR_COLORS[inhibitor])
        patch.set_alpha(0.7)

    ax1.set_ylabel('Inhibition Efficiency (%)', fontweight='bold')
    ax1.set_title('Distribution of IE% by Inhibitor', fontweight='bold')
    ax1.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% threshold')
    ax1.legend()

    # Violin plot
    ax2 = axes[1]

    plot_data = df[df['inhibitor_conc_mg_L'] > 0][['inhibitor_name', 'inhibition_efficiency_pct']].copy()
    plot_data['inhibitor_short'] = plot_data['inhibitor_name'].str.replace(' extract', '')

    colors_list = [INHIBITOR_COLORS[inh] for inh in inhibitors]
    palette = {inh.replace(' extract', ''): INHIBITOR_COLORS[inh] for inh in inhibitors}

    sns.violinplot(data=plot_data, x='inhibitor_short', y='inhibition_efficiency_pct',
                  hue='inhibitor_short', palette=palette, ax=ax2, legend=False)

    ax2.set_xlabel('Inhibitor', fontweight='bold')
    ax2.set_ylabel('Inhibition Efficiency (%)', fontweight='bold')
    ax2.set_title('IE% Distribution (Violin Plot)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_ie_distribution.png")
    plt.close()
    print(f"   Saved: 09_ie_distribution.png")


# =============================================================================
# 7. ML MODEL PERFORMANCE
# =============================================================================

def plot_model_predictions(df, models, df_train, df_val, df_test):
    """Plot ML model predictions vs actual values."""
    print("\n[GRAPH] ML model predictions...")

    if 'IE' not in models or 'preprocessor' not in models:
        print("   [SKIP] ML models not available")
        return

    if df_test is None:
        print("   [SKIP] Test data not available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Feature columns
    feature_cols = [
        "acid_molarity_M", "temperature_C", "immersion_time_h",
        "inhibitor_conc_mg_L", "log_conc_mg_L", "temp_conc_interaction",
        "acid_strength_norm", "acid_type_encoded", "acid", "inhibitor_name", "method"
    ]

    datasets = [('Train', df_train), ('Validation', df_val), ('Test', df_test)]

    for idx, (name, data) in enumerate(datasets):
        ax = axes[idx]

        if data is None or len(data) == 0:
            continue

        X = data[feature_cols]
        y_true = data['inhibition_efficiency_pct'].values

        X_prep = models['preprocessor'].transform(X)
        y_pred = models['IE'].predict(X_prep)
        y_pred = np.clip(y_pred, 0, 100)

        # Color by inhibitor
        for inhibitor, color in INHIBITOR_COLORS.items():
            mask = data['inhibitor_name'] == inhibitor
            if mask.sum() > 0:
                ax.scatter(y_true[mask], y_pred[mask], c=color,
                          marker=INHIBITOR_MARKERS[inhibitor],
                          s=60, alpha=0.7, label=inhibitor.split()[0])

        # Diagonal line
        lims = [min(y_true.min(), y_pred.min()) - 5, max(y_true.max(), y_pred.max()) + 5]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')

        # Metrics
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
        mae = np.mean(np.abs(y_true - y_pred))

        ax.set_xlabel('Actual IE (%)', fontweight='bold')
        ax.set_ylabel('Predicted IE (%)', fontweight='bold')
        ax.set_title(f'{name} Set\nR² = {r2:.4f}, MAE = {mae:.2f}%', fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    plt.suptitle('ML Model Performance: Predicted vs Actual IE%', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_model_predictions.png")
    plt.close()
    print(f"   Saved: 10_model_predictions.png")


def plot_feature_importance(models):
    """Plot feature importance from ML models."""
    print("\n[GRAPH] Feature importance...")

    # Try to load feature importance from file
    fi_path = Path("ml_models/best_model_IE_feature_importance.csv")

    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Sort by importance
        fi_df = fi_df.sort_values('importance', ascending=True)

        # Take top 15
        fi_df = fi_df.tail(15)

        colors = ['#3498db' if imp >= 0 else '#e74c3c' for imp in fi_df['importance']]

        ax.barh(fi_df['feature'], fi_df['importance'], color=colors, edgecolor='black')
        ax.set_xlabel('Permutation Importance', fontweight='bold')
        ax.set_ylabel('Feature', fontweight='bold')
        ax.set_title('Feature Importance for IE% Prediction\n(Permutation Importance)', fontweight='bold')
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "11_feature_importance.png")
        plt.close()
        print(f"   Saved: 11_feature_importance.png")
    else:
        print("   [SKIP] Feature importance file not found")


# =============================================================================
# 8. CORRELATION ANALYSIS
# =============================================================================

def plot_correlation_matrix(df):
    """Plot correlation matrix of numerical features."""
    print("\n[GRAPH] Correlation matrix...")

    # Select numerical columns
    num_cols = ['acid_molarity_M', 'temperature_C', 'immersion_time_h',
                'inhibitor_conc_mg_L', 'inhibition_efficiency_pct',
                'Ecorr_mV', 'Icorr_uA_cm2']

    # Filter to available columns
    num_cols = [col for col in num_cols if col in df.columns]

    # Calculate correlation matrix
    corr_matrix = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
               cmap='RdBu_r', center=0, ax=ax,
               square=True, linewidths=0.5,
               cbar_kws={'label': 'Correlation Coefficient'})

    ax.set_title('Correlation Matrix of Experimental Variables', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "12_correlation_matrix.png")
    plt.close()
    print(f"   Saved: 12_correlation_matrix.png")


# =============================================================================
# 9. SUMMARY INFOGRAPHIC
# =============================================================================

def plot_summary_infographic(df):
    """Create a summary infographic of the study."""
    print("\n[GRAPH] Summary infographic...")

    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Title and overview (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.text(0.5, 0.8, 'Green Corrosion Inhibitors', fontsize=18, fontweight='bold',
            ha='center', va='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.6, 'for Mild Steel in Acidic Medium', fontsize=14,
            ha='center', va='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.3, f'Dataset: {len(df)} experiments\n'
            f'Inhibitors: 3 plant extracts\n'
            f'Acids: H₂SO₄, HCl', fontsize=11,
            ha='center', va='center', transform=ax1.transAxes)

    # 2. Inhibitor icons (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    y_pos = [0.75, 0.45, 0.15]
    for i, (inhibitor, color) in enumerate(INHIBITOR_COLORS.items()):
        data = df[df['inhibitor_name'] == inhibitor]
        max_ie = data['inhibition_efficiency_pct'].max() if len(data) > 0 else 0
        ax2.add_patch(plt.Circle((0.15, y_pos[i]), 0.08, color=color, transform=ax2.transAxes))
        ax2.text(0.3, y_pos[i], f'{inhibitor}\nMax IE: {max_ie:.1f}%', fontsize=10,
                va='center', transform=ax2.transAxes)
    ax2.set_title('Inhibitors Studied', fontweight='bold', pad=10)

    # 3. Key statistics (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    stats_text = (
        f"Key Statistics:\n\n"
        f"• Total experiments: {len(df)}\n"
        f"• IE% range: {df['inhibition_efficiency_pct'].min():.1f}% - {df['inhibition_efficiency_pct'].max():.1f}%\n"
        f"• Mean IE%: {df['inhibition_efficiency_pct'].mean():.1f}%\n"
        f"• Conc. range: {df['inhibitor_conc_mg_L'].min():.0f} - {df['inhibitor_conc_mg_L'].max():.0f} mg/L\n"
        f"• Temp. range: {df['temperature_C'].min():.0f} - {df['temperature_C'].max():.0f}°C"
    )
    ax3.text(0.1, 0.5, stats_text, fontsize=11, va='center', transform=ax3.transAxes,
            family='monospace')
    ax3.set_title('Dataset Overview', fontweight='bold', pad=10)

    # 4. Concentration response (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    for inhibitor, color in INHIBITOR_COLORS.items():
        data = df[df['inhibitor_name'] == inhibitor]
        if len(data) > 0:
            grouped = data.groupby('inhibitor_conc_mg_L')['inhibition_efficiency_pct'].mean()
            ax4.plot(grouped.index, grouped.values, f'{INHIBITOR_MARKERS[inhibitor]}-',
                    color=color, markersize=6, label=inhibitor.split()[0])
    ax4.set_xlabel('Concentration (mg/L)')
    ax4.set_ylabel('IE (%)')
    ax4.set_title('Concentration Response', fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.set_ylim(-10, 105)

    # 5. Acid comparison (middle center)
    ax5 = fig.add_subplot(gs[1, 1])
    acids = ['H2SO4', 'HCl']
    for i, acid in enumerate(acids):
        acid_data = df[(df['acid'] == acid) & (df['inhibitor_conc_mg_L'] > 0)]
        means = [acid_data[acid_data['inhibitor_name'] == inh]['inhibition_efficiency_pct'].mean()
                for inh in INHIBITOR_COLORS.keys()]
        x = np.arange(len(INHIBITOR_COLORS)) + i * 0.35
        ax5.bar(x, means, 0.35, label=acid, color=ACID_COLORS[acid], alpha=0.8)
    ax5.set_xticks(np.arange(len(INHIBITOR_COLORS)) + 0.175)
    ax5.set_xticklabels([inh.split()[0] for inh in INHIBITOR_COLORS.keys()], fontsize=9)
    ax5.set_ylabel('Mean IE (%)')
    ax5.set_title('Acid Type Comparison', fontweight='bold')
    ax5.legend(fontsize=9)

    # 6. Distribution (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    for inhibitor, color in INHIBITOR_COLORS.items():
        data = df[(df['inhibitor_name'] == inhibitor) & (df['inhibitor_conc_mg_L'] > 0)]
        if len(data) > 0:
            ax6.hist(data['inhibition_efficiency_pct'], bins=10, alpha=0.5,
                    color=color, label=inhibitor.split()[0], edgecolor='black')
    ax6.set_xlabel('IE (%)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('IE% Distribution', fontweight='bold')
    ax6.legend(fontsize=8)

    # 7-9. Bottom row - ML results if available
    ax7 = fig.add_subplot(gs[2, :])

    # Load metrics if available
    metrics_path = Path("ml_models/best_model_IE_metrics.csv")
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)

        ax7.axis('off')

        # Create table
        table_data = []
        for _, row in metrics_df.iterrows():
            model_name = row.get('best_model', row.get('model_name', 'Unknown'))
            table_data.append([model_name, f"{row['val_r2']:.4f}",
                              f"{row['val_mae']:.2f}%", f"{row['test_r2']:.4f}",
                              f"{row['test_mae']:.2f}%"])

        table = ax7.table(cellText=table_data,
                         colLabels=['Model', 'Val R²', 'Val MAE', 'Test R²', 'Test MAE'],
                         loc='center', cellLoc='center',
                         colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Color header
        for i in range(5):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        ax7.set_title('ML Model Performance Summary', fontweight='bold', pad=20, fontsize=14)
    else:
        ax7.text(0.5, 0.5, 'ML model metrics not available', ha='center', va='center',
                transform=ax7.transAxes, fontsize=12)
        ax7.axis('off')

    plt.suptitle('Corrosion Inhibitor Study Summary', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / "13_summary_infographic.png")
    plt.close()
    print(f"   Saved: 13_summary_infographic.png")


# =============================================================================
# 10. 3D SURFACE PLOT
# =============================================================================

def plot_3d_surface(df, models):
    """Create 3D surface plot of IE% vs concentration and temperature."""
    print("\n[GRAPH] 3D surface plot...")

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 6))

    for idx, (inhibitor, color) in enumerate(INHIBITOR_COLORS.items()):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

        data = df[(df['inhibitor_name'] == inhibitor) & (df['inhibitor_conc_mg_L'] > 0)]

        if len(data) < 3:
            continue

        # Create meshgrid for surface
        conc_range = np.linspace(data['inhibitor_conc_mg_L'].min(),
                                  data['inhibitor_conc_mg_L'].max(), 20)
        temp_range = np.linspace(data['temperature_C'].min(),
                                  data['temperature_C'].max(), 20)

        C, T = np.meshgrid(conc_range, temp_range)

        # Interpolate IE values
        from scipy.interpolate import griddata
        points = data[['inhibitor_conc_mg_L', 'temperature_C']].values
        values = data['inhibition_efficiency_pct'].values

        IE = griddata(points, values, (C, T), method='linear')

        # Plot surface
        surf = ax.plot_surface(C, T, IE, cmap='viridis', alpha=0.8,
                               linewidth=0, antialiased=True)

        # Scatter actual data points
        ax.scatter(data['inhibitor_conc_mg_L'], data['temperature_C'],
                  data['inhibition_efficiency_pct'], c='red', s=30, alpha=0.8)

        ax.set_xlabel('Conc (mg/L)', fontsize=10)
        ax.set_ylabel('Temp (°C)', fontsize=10)
        ax.set_zlabel('IE (%)', fontsize=10)
        ax.set_title(inhibitor.replace(' extract', ''), fontweight='bold')

    plt.suptitle('3D Surface: IE% vs Concentration and Temperature',
                fontweight='bold', fontsize=14, y=0.95)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "14_3d_surface.png")
    plt.close()
    print(f"   Saved: 14_3d_surface.png")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate study graphs")
    parser.add_argument('--category', '-c', type=str, default='all',
                       help='Graph category to generate')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available categories')
    args = parser.parse_args()

    categories = {
        'concentration': ['plot_concentration_vs_ie', 'plot_concentration_comparison',
                         'plot_concentration_log_scale'],
        'acid': ['plot_acid_comparison'],
        'temperature': ['plot_temperature_effect'],
        'adsorption': ['plot_adsorption_isotherm'],
        'electrochemical': ['plot_electrochemical_data'],
        'comparison': ['plot_max_ie_comparison', 'plot_ie_distribution'],
        'model': ['plot_model_predictions', 'plot_feature_importance'],
        'correlation': ['plot_correlation_matrix'],
        'summary': ['plot_summary_infographic'],
        '3d': ['plot_3d_surface'],
    }

    if args.list:
        print("\nAvailable graph categories:")
        for cat, funcs in categories.items():
            print(f"  {cat}: {len(funcs)} graphs")
        print("\nUse --category <name> to generate specific category")
        print("Use --category all to generate all graphs")
        return

    print("=" * 70)
    print("COMPREHENSIVE GRAPH GENERATION FOR CORROSION INHIBITOR STUDY")
    print("=" * 70)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\n[INFO] Output directory: {OUTPUT_DIR}")

    # Load data
    df, df_train, df_val, df_test, df_echem = load_data()
    models = load_models()

    # Generate graphs
    print("\n" + "=" * 70)
    print("GENERATING GRAPHS")
    print("=" * 70)

    if args.category == 'all':
        # Generate all graphs
        plot_concentration_vs_ie(df)
        plot_concentration_comparison(df)
        plot_concentration_log_scale(df)
        plot_acid_comparison(df)
        plot_temperature_effect(df)
        plot_adsorption_isotherm(df)
        plot_electrochemical_data(df, df_echem)
        plot_max_ie_comparison(df)
        plot_ie_distribution(df)
        plot_model_predictions(df, models, df_train, df_val, df_test)
        plot_feature_importance(models)
        plot_correlation_matrix(df)
        plot_summary_infographic(df)
        plot_3d_surface(df, models)
    else:
        # Generate specific category
        if args.category not in categories:
            print(f"[ERROR] Unknown category: {args.category}")
            print(f"Available: {list(categories.keys())}")
            return

        for func_name in categories[args.category]:
            func = globals()[func_name]
            if 'models' in func.__code__.co_varnames:
                if 'df_train' in func.__code__.co_varnames:
                    func(df, models, df_train, df_val, df_test)
                else:
                    func(df, models)
            elif 'df_echem' in func.__code__.co_varnames:
                func(df, df_echem)
            else:
                func(df)

    # Summary
    print("\n" + "=" * 70)
    print("GRAPH GENERATION COMPLETE")
    print("=" * 70)

    n_files = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"\n[INFO] Generated {n_files} graphs in: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"   - {f.name}")


if __name__ == "__main__":
    main()
