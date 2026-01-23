#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thesis Figure Generator for Corrosion Inhibitor Study
======================================================

Generates publication-quality figures for thesis/research papers focusing on
the three green inhibitors: Curry leaf, Spinach leaf, and Peanut shell extracts.

Figures Generated:
1. Concentration-Response Curves (all inhibitors compared)
2. Temperature Effect Analysis (Peanut shell)
3. Immersion Time Effect Analysis (Spinach & Curry leaf)
4. 3D Surface Plots (Concentration-Temperature-IE interaction)
5. Comparative Bar Charts (Maximum IE% comparison)
6. Predicted vs Actual with Experimental Data Overlay
7. Langmuir Adsorption Isotherm Analysis
8. Heatmap of Optimal Conditions
9. Feature Importance Analysis
10. Model Performance Summary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import joblib
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
})

# Color palette for the 3 inhibitors
INHIBITOR_COLORS = {
    'Curry leaf extract': '#2E86AB',      # Blue
    'Spinach leaf extract': '#28A745',     # Green
    'Peanut shell extract': '#D4A574',     # Brown/Tan
}

INHIBITOR_MARKERS = {
    'Curry leaf extract': 'o',
    'Spinach leaf extract': 's',
    'Peanut shell extract': '^',
}

INHIBITOR_SCIENTIFIC = {
    'Curry leaf extract': 'Murraya koenigii',
    'Spinach leaf extract': 'Spinacia oleracea',
    'Peanut shell extract': 'Arachis hypogaea',
}

# Output directory
OUTPUT_DIR = Path("thesis_figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_model_and_data():
    """Load trained model and experimental data."""
    print("Loading model and data...")

    model = joblib.load("ml_models/best_model_model.pkl")
    preprocessor = joblib.load("ml_models/best_model_preprocessor.pkl")
    data = pd.read_csv("preprocessed_data/full_processed_data.csv")

    print(f"  Loaded {len(data)} experimental data points")
    print(f"  Inhibitors: {data['inhibitor_name'].unique().tolist()}")

    return model, preprocessor, data


def predict_ie(model, preprocessor, conditions_list):
    """Predict IE for multiple conditions."""
    df = pd.DataFrame(conditions_list)

    # Add engineered features
    df["log_conc_mg_L"] = np.log10(df["inhibitor_conc_mg_L"] + 1e-3)
    df["temp_conc_interaction"] = df["temperature_C"] * df["inhibitor_conc_mg_L"] / 1000.0
    df["acid_strength_norm"] = df["acid_molarity_M"] / 0.5

    feature_cols = [
        "acid_molarity_M", "temperature_C", "immersion_time_h",
        "inhibitor_conc_mg_L", "log_conc_mg_L", "temp_conc_interaction",
        "acid_strength_norm", "inhibitor_name", "method"
    ]

    X = df[feature_cols]
    X_prep = preprocessor.transform(X)
    predictions = model.predict(X_prep)

    return np.clip(predictions, 0, 100)


def fig1_concentration_response_comparison(model, preprocessor, data):
    """
    Figure 1: Concentration-Response Curves for all 3 inhibitors
    Shows predicted curves with experimental data points overlaid
    """
    print("\nGenerating Figure 1: Concentration-Response Comparison...")

    fig, ax = plt.subplots(figsize=(10, 7))

    conc_range = np.logspace(0, 3.5, 100)  # 1 to ~3000 mg/L

    # Conditions for each inhibitor (using their typical experimental conditions)
    inhibitor_conditions = {
        'Curry leaf extract': {'acid_molarity_M': 2.0, 'temperature_C': 25.0,
                               'immersion_time_h': 1.0, 'method': 'Weight loss'},
        'Spinach leaf extract': {'acid_molarity_M': 0.1, 'temperature_C': 25.0,
                                  'immersion_time_h': 24.0, 'method': 'Weight loss'},
        'Peanut shell extract': {'acid_molarity_M': 0.5, 'temperature_C': 25.0,
                                  'immersion_time_h': 12.0, 'method': 'Weight loss'},
    }

    for inhibitor, base_cond in inhibitor_conditions.items():
        # Generate predictions
        conditions = []
        for conc in conc_range:
            cond = base_cond.copy()
            cond['inhibitor_name'] = inhibitor
            cond['inhibitor_conc_mg_L'] = conc
            conditions.append(cond)

        predictions = predict_ie(model, preprocessor, conditions)

        # Plot predicted curve
        ax.plot(conc_range, predictions, '-',
                color=INHIBITOR_COLORS[inhibitor],
                linewidth=2.5,
                label=f'{inhibitor} (predicted)')

        # Overlay experimental data points
        exp_data = data[(data['inhibitor_name'] == inhibitor) &
                        (data['inhibitor_conc_mg_L'] > 0)]

        ax.scatter(exp_data['inhibitor_conc_mg_L'],
                  exp_data['inhibition_efficiency_pct'],
                  marker=INHIBITOR_MARKERS[inhibitor],
                  s=80,
                  color=INHIBITOR_COLORS[inhibitor],
                  edgecolor='white',
                  linewidth=1.5,
                  alpha=0.8,
                  label=f'{inhibitor} (experimental)',
                  zorder=5)

    ax.set_xscale('log')
    ax.set_xlabel('Inhibitor Concentration (mg/L)', fontweight='bold')
    ax.set_ylabel('Inhibition Efficiency (%)', fontweight='bold')
    ax.set_title('Concentration-Response Curves for Green Corrosion Inhibitors\nin H₂SO₄ Environment on Mild Steel',
                 fontweight='bold', pad=15)

    ax.set_xlim(1, 5000)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    # Add annotations
    ax.annotate('', xy=(3000, 80), xytext=(100, 80),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(500, 83, 'Increasing protection', ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_concentration_response_comparison.png")
    plt.savefig(OUTPUT_DIR / "fig1_concentration_response_comparison.pdf")
    plt.close()
    print("  Saved: fig1_concentration_response_comparison.png/pdf")


def fig2_temperature_effect(model, preprocessor, data):
    """
    Figure 2: Temperature Effect on Inhibition Efficiency
    Focuses on Peanut shell extract which has temperature data (25-40°C)
    """
    print("\nGenerating Figure 2: Temperature Effect Analysis...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: IE vs Temperature for Peanut shell at 300 mg/L
    temps = np.linspace(20, 50, 50)

    conditions = []
    for temp in temps:
        conditions.append({
            'inhibitor_name': 'Peanut shell extract',
            'acid_molarity_M': 0.5,
            'temperature_C': temp,
            'immersion_time_h': 12.0,
            'inhibitor_conc_mg_L': 300.0,
            'method': 'Weight loss'
        })

    predictions = predict_ie(model, preprocessor, conditions)

    ax1.plot(temps, predictions, 'b-', linewidth=2.5, label='Predicted (300 mg/L)')

    # Overlay experimental data
    peanut_data = data[(data['inhibitor_name'] == 'Peanut shell extract') &
                       (data['inhibitor_conc_mg_L'] == 300)]
    ax1.scatter(peanut_data['temperature_C'],
               peanut_data['inhibition_efficiency_pct'],
               s=100, color=INHIBITOR_COLORS['Peanut shell extract'],
               edgecolor='black', linewidth=1.5, zorder=5,
               label='Experimental data')

    ax1.set_xlabel('Temperature (°C)', fontweight='bold')
    ax1.set_ylabel('Inhibition Efficiency (%)', fontweight='bold')
    ax1.set_title('(a) Temperature Effect on Peanut Shell Extract\n(300 mg/L in 0.5M H₂SO₄)',
                  fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlim(20, 50)
    ax1.set_ylim(50, 100)
    ax1.grid(True, alpha=0.3)

    # Add trend annotation
    ax1.annotate('Decreasing IE\nwith temperature', xy=(38, 70), fontsize=9,
                 ha='center', color='gray',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right plot: Bar chart comparing IE at different temperatures
    temp_labels = ['25°C', '30°C', '35°C', '40°C']
    exp_ie = peanut_data.sort_values('temperature_C')['inhibition_efficiency_pct'].values

    bars = ax2.bar(temp_labels, exp_ie, color=INHIBITOR_COLORS['Peanut shell extract'],
                   edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, exp_ie):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)

    ax2.set_xlabel('Temperature', fontweight='bold')
    ax2.set_ylabel('Inhibition Efficiency (%)', fontweight='bold')
    ax2.set_title('(b) Experimental IE at Different Temperatures\n(Peanut Shell Extract, 300 mg/L)',
                  fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_temperature_effect.png")
    plt.savefig(OUTPUT_DIR / "fig2_temperature_effect.pdf")
    plt.close()
    print("  Saved: fig2_temperature_effect.png/pdf")


def fig3_immersion_time_effect(model, preprocessor, data):
    """
    Figure 3: Immersion Time Effect on Inhibition Efficiency
    Shows how IE changes with exposure time for different inhibitors
    """
    print("\nGenerating Figure 3: Immersion Time Effect...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Spinach leaf - long immersion times (24-168h)
    spinach_data = data[data['inhibitor_name'] == 'Spinach leaf extract']
    spinach_data = spinach_data.sort_values('immersion_time_h')

    ax1.plot(spinach_data['immersion_time_h'],
             spinach_data['inhibition_efficiency_pct'],
             'o-', color=INHIBITOR_COLORS['Spinach leaf extract'],
             markersize=10, linewidth=2.5, markeredgecolor='white',
             markeredgewidth=2, label='Experimental data')

    # Add trend line
    z = np.polyfit(spinach_data['immersion_time_h'],
                   spinach_data['inhibition_efficiency_pct'], 1)
    p = np.poly1d(z)
    time_range = np.linspace(20, 180, 50)
    ax1.plot(time_range, p(time_range), '--', color='gray',
             linewidth=1.5, alpha=0.7, label='Linear trend')

    ax1.set_xlabel('Immersion Time (hours)', fontweight='bold')
    ax1.set_ylabel('Inhibition Efficiency (%)', fontweight='bold')
    ax1.set_title('(a) Spinach Leaf Extract\n(500 mg/L in 0.1M H₂SO₄)', fontweight='bold')
    ax1.legend(loc='lower left')
    ax1.set_xlim(0, 180)
    ax1.set_ylim(70, 95)
    ax1.grid(True, alpha=0.3)

    # Calculate degradation rate
    slope = z[0]
    ax1.text(100, 90, f'Degradation rate:\n{abs(slope):.3f}%/hour',
             fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Right plot: Curry leaf - short immersion times (1-5h)
    curry_data = data[(data['inhibitor_name'] == 'Curry leaf extract') &
                      (data['method'] == 'Weight loss') &
                      (data['inhibitor_conc_mg_L'] > 0)]

    # Group by concentration and time
    for conc in [1000, 2000, 3000]:
        conc_data = curry_data[curry_data['inhibitor_conc_mg_L'] == conc]
        conc_data = conc_data.sort_values('immersion_time_h')
        if len(conc_data) > 0:
            ax2.plot(conc_data['immersion_time_h'],
                    conc_data['inhibition_efficiency_pct'],
                    'o-', markersize=8, linewidth=2,
                    label=f'{int(conc)} mg/L')

    ax2.set_xlabel('Immersion Time (hours)', fontweight='bold')
    ax2.set_ylabel('Inhibition Efficiency (%)', fontweight='bold')
    ax2.set_title('(b) Curry Leaf Extract at Different Concentrations\n(in 2M H₂SO₄)',
                  fontweight='bold')
    ax2.legend(loc='upper right', title='Concentration')
    ax2.set_xlim(0, 6)
    ax2.set_ylim(40, 90)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_immersion_time_effect.png")
    plt.savefig(OUTPUT_DIR / "fig3_immersion_time_effect.pdf")
    plt.close()
    print("  Saved: fig3_immersion_time_effect.png/pdf")


def fig4_3d_surface_plots(model, preprocessor):
    """
    Figure 4: 3D Surface Plots showing IE as function of concentration and temperature
    """
    print("\nGenerating Figure 4: 3D Surface Plots...")

    fig = plt.figure(figsize=(14, 5))

    inhibitors = ['Curry leaf extract', 'Peanut shell extract', 'Spinach leaf extract']
    base_conditions = [
        {'acid_molarity_M': 2.0, 'immersion_time_h': 1.0, 'method': 'Weight loss'},
        {'acid_molarity_M': 0.5, 'immersion_time_h': 12.0, 'method': 'Weight loss'},
        {'acid_molarity_M': 0.1, 'immersion_time_h': 24.0, 'method': 'Weight loss'},
    ]

    conc_range = np.linspace(100, 3000, 25)
    temp_range = np.linspace(25, 45, 20)
    CONC, TEMP = np.meshgrid(conc_range, temp_range)

    for idx, (inhibitor, base_cond) in enumerate(zip(inhibitors, base_conditions)):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')

        # Generate predictions for the grid
        conditions = []
        for c, t in zip(CONC.flatten(), TEMP.flatten()):
            cond = base_cond.copy()
            cond['inhibitor_name'] = inhibitor
            cond['inhibitor_conc_mg_L'] = c
            cond['temperature_C'] = t
            conditions.append(cond)

        predictions = predict_ie(model, preprocessor, conditions)
        IE = predictions.reshape(CONC.shape)

        # Plot surface
        surf = ax.plot_surface(CONC, TEMP, IE, cmap=cm.viridis,
                               linewidth=0, antialiased=True, alpha=0.8)

        ax.set_xlabel('Conc. (mg/L)', fontsize=9)
        ax.set_ylabel('Temp. (°C)', fontsize=9)
        ax.set_zlabel('IE (%)', fontsize=9)
        ax.set_title(f'{inhibitor.replace(" extract", "")}\n({INHIBITOR_SCIENTIFIC[inhibitor]})',
                     fontsize=10, fontweight='bold')
        ax.set_zlim(0, 100)
        ax.view_init(elev=25, azim=45)

    plt.suptitle('3D Response Surfaces: Inhibition Efficiency vs Concentration and Temperature',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_3d_surface_plots.png")
    plt.savefig(OUTPUT_DIR / "fig4_3d_surface_plots.pdf")
    plt.close()
    print("  Saved: fig4_3d_surface_plots.png/pdf")


def fig5_comparative_bar_chart(data):
    """
    Figure 5: Comparative Bar Chart of Maximum IE% for each inhibitor
    """
    print("\nGenerating Figure 5: Comparative Performance Bar Chart...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Maximum IE achieved
    inhibitors = ['Curry leaf extract', 'Spinach leaf extract', 'Peanut shell extract']
    max_ie = []
    mean_ie = []
    std_ie = []

    for inh in inhibitors:
        inh_data = data[(data['inhibitor_name'] == inh) & (data['inhibitor_conc_mg_L'] > 0)]
        max_ie.append(inh_data['inhibition_efficiency_pct'].max())
        mean_ie.append(inh_data['inhibition_efficiency_pct'].mean())
        std_ie.append(inh_data['inhibition_efficiency_pct'].std())

    x = np.arange(len(inhibitors))
    width = 0.35

    bars1 = ax1.bar(x - width/2, max_ie, width, label='Maximum IE',
                    color=[INHIBITOR_COLORS[i] for i in inhibitors],
                    edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, mean_ie, width, label='Mean IE',
                    color=[INHIBITOR_COLORS[i] for i in inhibitors],
                    edgecolor='black', linewidth=1.5, alpha=0.6)

    # Add error bars for mean
    ax1.errorbar(x + width/2, mean_ie, yerr=std_ie, fmt='none',
                 color='black', capsize=5, capthick=2)

    # Add value labels
    for bar, val in zip(bars1, max_ie):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)

    ax1.set_ylabel('Inhibition Efficiency (%)', fontweight='bold')
    ax1.set_title('(a) Maximum and Mean IE for Each Inhibitor', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([i.replace(' extract', '\nextract') for i in inhibitors])
    ax1.legend()
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add 80% threshold line
    ax1.axhline(y=80, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(2.3, 81, '80% threshold', fontsize=8, color='red')

    # Right: Optimal concentration for each inhibitor
    optimal_conc = []
    for inh in inhibitors:
        inh_data = data[(data['inhibitor_name'] == inh) & (data['inhibitor_conc_mg_L'] > 0)]
        max_idx = inh_data['inhibition_efficiency_pct'].idxmax()
        optimal_conc.append(inh_data.loc[max_idx, 'inhibitor_conc_mg_L'])

    bars3 = ax2.bar(x, optimal_conc, width=0.6,
                    color=[INHIBITOR_COLORS[i] for i in inhibitors],
                    edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars3, optimal_conc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{int(val)} mg/L', ha='center', fontweight='bold', fontsize=10)

    ax2.set_ylabel('Concentration (mg/L)', fontweight='bold')
    ax2.set_title('(b) Optimal Concentration for Maximum IE', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([i.replace(' extract', '\nextract') for i in inhibitors])
    ax2.set_ylim(0, 3500)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_comparative_performance.png")
    plt.savefig(OUTPUT_DIR / "fig5_comparative_performance.pdf")
    plt.close()
    print("  Saved: fig5_comparative_performance.png/pdf")


def fig6_predicted_vs_actual(model, preprocessor, data):
    """
    Figure 6: Predicted vs Actual IE with 1:1 line and statistics
    """
    print("\nGenerating Figure 6: Predicted vs Actual Scatter...")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Get predictions for all experimental data
    exp_data = data[data['inhibitor_conc_mg_L'] > 0].copy()

    conditions = []
    for _, row in exp_data.iterrows():
        conditions.append({
            'inhibitor_name': row['inhibitor_name'],
            'acid_molarity_M': row['acid_molarity_M'],
            'temperature_C': row['temperature_C'],
            'immersion_time_h': row['immersion_time_h'],
            'inhibitor_conc_mg_L': row['inhibitor_conc_mg_L'],
            'method': row['method']
        })

    predictions = predict_ie(model, preprocessor, conditions)
    actual = exp_data['inhibition_efficiency_pct'].values

    # Plot by inhibitor
    inhibitors = exp_data['inhibitor_name'].unique()
    for inh in inhibitors:
        mask = exp_data['inhibitor_name'] == inh
        ax.scatter(actual[mask], predictions[mask],
                  c=INHIBITOR_COLORS[inh],
                  marker=INHIBITOR_MARKERS[inh],
                  s=100, alpha=0.7, edgecolor='white',
                  linewidth=1.5, label=inh)

    # Add 1:1 line
    lims = [0, 105]
    ax.plot(lims, lims, 'k-', linewidth=2, label='Perfect prediction (1:1)')

    # Add ±10% bands
    ax.fill_between(lims, [l-10 for l in lims], [l+10 for l in lims],
                    alpha=0.1, color='gray', label='±10% band')

    # Calculate statistics
    r2 = 1 - np.sum((actual - predictions)**2) / np.sum((actual - np.mean(actual))**2)
    mae = np.mean(np.abs(actual - predictions))
    rmse = np.sqrt(np.mean((actual - predictions)**2))

    # Add statistics box
    stats_text = f'R² = {r2:.3f}\nMAE = {mae:.1f}%\nRMSE = {rmse:.1f}%'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('Actual Inhibition Efficiency (%)', fontweight='bold')
    ax.set_ylabel('Predicted Inhibition Efficiency (%)', fontweight='bold')
    ax.set_title('Model Validation: Predicted vs Actual IE\nfor Green Corrosion Inhibitors',
                 fontweight='bold', pad=15)
    ax.legend(loc='lower right')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_predicted_vs_actual.png")
    plt.savefig(OUTPUT_DIR / "fig6_predicted_vs_actual.pdf")
    plt.close()
    print("  Saved: fig6_predicted_vs_actual.png/pdf")


def fig7_langmuir_isotherm(data):
    """
    Figure 7: Langmuir Adsorption Isotherm Analysis
    """
    print("\nGenerating Figure 7: Langmuir Adsorption Isotherm...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    inhibitors = ['Curry leaf extract', 'Spinach leaf extract', 'Peanut shell extract']

    for idx, inhibitor in enumerate(inhibitors):
        ax = axes[idx]

        # Get data for this inhibitor
        inh_data = data[(data['inhibitor_name'] == inhibitor) &
                        (data['inhibitor_conc_mg_L'] > 0) &
                        (data['surface_coverage'] > 0) &
                        (data['surface_coverage'] < 1)]

        # Check if we have enough unique concentration values
        unique_conc = inh_data['inhibitor_conc_mg_L'].nunique()

        if len(inh_data) < 3 or unique_conc < 2:
            ax.text(0.5, 0.5, 'Insufficient data\nfor isotherm fitting\n(single concentration)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xlabel('Concentration (mg/L)', fontweight='bold')
            ax.set_ylabel('C/θ (mg/L)', fontweight='bold')
            ax.set_title(f'({chr(97+idx)}) {inhibitor.replace(" extract", "")}\n({INHIBITOR_SCIENTIFIC[inhibitor]})',
                         fontweight='bold')
            ax.set_xlim(0, 3500)
            ax.set_ylim(0, 6000)
            continue

        # Langmuir isotherm: C/theta = 1/Kads + C
        C = inh_data['inhibitor_conc_mg_L'].values
        theta = inh_data['surface_coverage'].values
        C_over_theta = C / theta

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(C, C_over_theta)
        Kads = 1 / intercept if intercept > 0 else np.nan

        # Plot
        ax.scatter(C, C_over_theta, s=80, c=INHIBITOR_COLORS[inhibitor],
                  edgecolor='black', linewidth=1.5, zorder=5)

        # Fitted line
        C_fit = np.linspace(C.min(), C.max(), 100)
        C_over_theta_fit = slope * C_fit + intercept
        ax.plot(C_fit, C_over_theta_fit, '--', color='gray', linewidth=2)

        # Add statistics
        stats_text = f'R² = {r_value**2:.4f}\nKads = {Kads:.2e} L/mg\nSlope = {slope:.4f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('Concentration (mg/L)', fontweight='bold')
        ax.set_ylabel('C/θ (mg/L)', fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {inhibitor.replace(" extract", "")}\n({INHIBITOR_SCIENTIFIC[inhibitor]})',
                     fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Langmuir Adsorption Isotherm Analysis', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig7_langmuir_isotherm.png")
    plt.savefig(OUTPUT_DIR / "fig7_langmuir_isotherm.pdf")
    plt.close()
    print("  Saved: fig7_langmuir_isotherm.png/pdf")


def fig8_heatmap_optimal_conditions(model, preprocessor):
    """
    Figure 8: Heatmap showing IE across concentration and temperature
    """
    print("\nGenerating Figure 8: Optimal Conditions Heatmap...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    inhibitors = ['Curry leaf extract', 'Spinach leaf extract', 'Peanut shell extract']
    base_conditions = [
        {'acid_molarity_M': 2.0, 'immersion_time_h': 1.0, 'method': 'Weight loss'},
        {'acid_molarity_M': 0.1, 'immersion_time_h': 24.0, 'method': 'Weight loss'},
        {'acid_molarity_M': 0.5, 'immersion_time_h': 12.0, 'method': 'Weight loss'},
    ]

    conc_values = [100, 300, 500, 1000, 1500, 2000, 2500, 3000]
    temp_values = [25, 30, 35, 40]

    for idx, (inhibitor, base_cond) in enumerate(zip(inhibitors, base_conditions)):
        ax = axes[idx]

        # Generate predictions for the grid
        IE_matrix = np.zeros((len(temp_values), len(conc_values)))

        for i, temp in enumerate(temp_values):
            for j, conc in enumerate(conc_values):
                cond = base_cond.copy()
                cond['inhibitor_name'] = inhibitor
                cond['inhibitor_conc_mg_L'] = conc
                cond['temperature_C'] = temp
                predictions = predict_ie(model, preprocessor, [cond])
                IE_matrix[i, j] = predictions[0]

        # Create heatmap
        im = ax.imshow(IE_matrix, cmap='RdYlGn', aspect='auto',
                       vmin=0, vmax=100, origin='lower')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('IE (%)', fontsize=10)

        # Set labels
        ax.set_xticks(np.arange(len(conc_values)))
        ax.set_yticks(np.arange(len(temp_values)))
        ax.set_xticklabels(conc_values, fontsize=8)
        ax.set_yticklabels(temp_values)

        # Add value annotations
        for i in range(len(temp_values)):
            for j in range(len(conc_values)):
                text_color = 'white' if IE_matrix[i, j] < 50 else 'black'
                ax.text(j, i, f'{IE_matrix[i, j]:.0f}',
                       ha='center', va='center', fontsize=8, color=text_color)

        ax.set_xlabel('Concentration (mg/L)', fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontweight='bold')
        ax.set_title(f'({chr(97+idx)}) {inhibitor.replace(" extract", "")}', fontweight='bold')

    plt.suptitle('Predicted Inhibition Efficiency Heatmaps\n(Concentration × Temperature)',
                 fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig8_heatmap_optimal_conditions.png")
    plt.savefig(OUTPUT_DIR / "fig8_heatmap_optimal_conditions.pdf")
    plt.close()
    print("  Saved: fig8_heatmap_optimal_conditions.png/pdf")


def fig9_experimental_data_summary(data):
    """
    Figure 9: Summary of Experimental Data Distribution
    """
    print("\nGenerating Figure 9: Experimental Data Summary...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) IE distribution by inhibitor - violin plot
    ax1 = axes[0, 0]
    inhibitors = ['Curry leaf extract', 'Spinach leaf extract', 'Peanut shell extract']
    exp_data = data[(data['inhibitor_conc_mg_L'] > 0)]

    parts = ax1.violinplot([exp_data[exp_data['inhibitor_name'] == inh]['inhibition_efficiency_pct'].values
                           for inh in inhibitors],
                          positions=range(len(inhibitors)), showmeans=True, showmedians=True)

    # Color the violins
    for i, (pc, inh) in enumerate(zip(parts['bodies'], inhibitors)):
        pc.set_facecolor(INHIBITOR_COLORS[inh])
        pc.set_alpha(0.7)

    ax1.set_xticks(range(len(inhibitors)))
    ax1.set_xticklabels([i.replace(' extract', '\nextract') for i in inhibitors])
    ax1.set_ylabel('Inhibition Efficiency (%)', fontweight='bold')
    ax1.set_title('(a) IE Distribution by Inhibitor', fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis='y')

    # (b) Data points by experimental condition
    ax2 = axes[0, 1]
    condition_counts = exp_data.groupby(['inhibitor_name', 'method']).size().unstack(fill_value=0)
    condition_counts.plot(kind='bar', ax=ax2, color=['#3498db', '#e74c3c'],
                          edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('')
    ax2.set_ylabel('Number of Data Points', fontweight='bold')
    ax2.set_title('(b) Data Points by Method', fontweight='bold')
    ax2.set_xticklabels([i.replace(' extract', '') for i in condition_counts.index],
                        rotation=45, ha='right')
    ax2.legend(title='Method')
    ax2.grid(True, alpha=0.3, axis='y')

    # (c) Concentration distribution
    ax3 = axes[1, 0]
    for inh in inhibitors:
        inh_data = exp_data[exp_data['inhibitor_name'] == inh]
        ax3.hist(inh_data['inhibitor_conc_mg_L'], bins=10, alpha=0.6,
                label=inh.replace(' extract', ''), color=INHIBITOR_COLORS[inh],
                edgecolor='black', linewidth=1)
    ax3.set_xlabel('Inhibitor Concentration (mg/L)', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('(c) Concentration Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # (d) Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create summary table
    summary_data = []
    for inh in inhibitors:
        inh_data = exp_data[exp_data['inhibitor_name'] == inh]
        summary_data.append([
            inh.replace(' extract', ''),
            len(inh_data),
            f"{inh_data['inhibition_efficiency_pct'].mean():.1f} ± {inh_data['inhibition_efficiency_pct'].std():.1f}",
            f"{inh_data['inhibition_efficiency_pct'].max():.1f}",
            f"{inh_data['inhibitor_conc_mg_L'].min():.0f} - {inh_data['inhibitor_conc_mg_L'].max():.0f}"
        ])

    table = ax4.table(cellText=summary_data,
                      colLabels=['Inhibitor', 'N', 'Mean IE ± SD (%)', 'Max IE (%)', 'Conc. Range (mg/L)'],
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#2C3E50')
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')

    ax4.set_title('(d) Summary Statistics', fontweight='bold', y=0.95)

    plt.suptitle('Experimental Data Overview for Green Corrosion Inhibitors',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig9_experimental_data_summary.png")
    plt.savefig(OUTPUT_DIR / "fig9_experimental_data_summary.pdf")
    plt.close()
    print("  Saved: fig9_experimental_data_summary.png/pdf")


def fig10_model_performance_summary():
    """
    Figure 10: Model Performance Summary from Training
    """
    print("\nGenerating Figure 10: Model Performance Summary...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Model comparison data (from training report)
    models = ['Ridge', 'HistGB', 'GradientBoost', 'Ensemble', 'RandomForest', 'SVR']
    val_mae = [16.94, 41.36, 41.96, 42.06, 42.87, 48.37]
    cv_r2 = [0.17, -0.001, -0.009, -0.003, -0.001, -0.12]

    # (a) Validation MAE comparison
    colors = ['#28A745' if mae == min(val_mae) else '#3498db' for mae in val_mae]
    bars = ax1.bar(models, val_mae, color=colors, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, val_mae):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)

    ax1.set_ylabel('Validation MAE (%)', fontweight='bold')
    ax1.set_title('(a) Model Comparison: Validation MAE', fontweight='bold')
    ax1.set_ylim(0, 60)
    ax1.axhline(y=20, color='red', linestyle='--', alpha=0.5)
    ax1.text(5.2, 21, 'Target', fontsize=8, color='red')
    ax1.grid(True, alpha=0.3, axis='y')

    # (b) Cross-validation R² scores
    colors2 = ['#28A745' if r2 == max(cv_r2) else '#e74c3c' if r2 < 0 else '#3498db'
               for r2 in cv_r2]
    bars2 = ax2.bar(models, cv_r2, color=colors2, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars2, cv_r2):
        y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.03
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:.3f}', ha='center', fontweight='bold', fontsize=9)

    ax2.set_ylabel('Cross-Validation R²', fontweight='bold')
    ax2.set_title('(b) Model Comparison: CV R² Score', fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylim(-0.2, 0.3)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Machine Learning Model Performance Comparison',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig10_model_performance_summary.png")
    plt.savefig(OUTPUT_DIR / "fig10_model_performance_summary.pdf")
    plt.close()
    print("  Saved: fig10_model_performance_summary.png/pdf")


def generate_all_figures():
    """Generate all thesis figures."""
    print("="*70)
    print("THESIS FIGURE GENERATOR")
    print("Corrosion Inhibitor Study - Green Inhibitors in H2SO4")
    print("="*70)

    # Load model and data
    model, preprocessor, data = load_model_and_data()

    # Generate all figures
    fig1_concentration_response_comparison(model, preprocessor, data)
    fig2_temperature_effect(model, preprocessor, data)
    fig3_immersion_time_effect(model, preprocessor, data)
    fig4_3d_surface_plots(model, preprocessor)
    fig5_comparative_bar_chart(data)
    fig6_predicted_vs_actual(model, preprocessor, data)
    fig7_langmuir_isotherm(data)
    fig8_heatmap_optimal_conditions(model, preprocessor)
    fig9_experimental_data_summary(data)
    fig10_model_performance_summary()

    print("\n" + "="*70)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")
    print("\nPDF versions also saved for high-quality printing.")
    print("="*70)


if __name__ == "__main__":
    generate_all_figures()
