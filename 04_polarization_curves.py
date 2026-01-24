#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polarization Curve Generator for Corrosion Inhibitor Prediction
================================================================

Generates Tafel-style polarization curves using the Butler-Volmer equation
and trained ML models for Ecorr and Icorr prediction.

The Butler-Volmer equation:
    i = i_corr * (exp(2.303*(E-E_corr)/ba) - exp(-2.303*(E-E_corr)/bc))

Where:
    - i = current density (uA/cm2)
    - i_corr = corrosion current density
    - E = applied potential (mV vs SCE)
    - E_corr = corrosion potential
    - ba = anodic Tafel slope (mV/decade)
    - bc = cathodic Tafel slope (mV/decade)

Usage:
    # Interactive mode
    python 04_polarization_curves.py

    # Generate curve for specific inhibitor
    python 04_polarization_curves.py --inhibitor "Aloe vera extract" --conc 500

    # Compare multiple concentrations
    python 04_polarization_curves.py --mode compare --inhibitor "Curry leaf extract"

    # Generate curves for all inhibitors
    python 04_polarization_curves.py --mode all --output polarization_all.png
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import joblib
import warnings
warnings.filterwarnings('ignore')


# Configuration
MODEL_DIR = Path("ml_models")
FIGURES_DIR = Path("ml_figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Default Tafel slopes from literature (mV/decade)
# These are typical values for mild steel in acidic solutions
DEFAULT_TAFEL_SLOPES = {
    "H2SO4": {"ba": 60, "bc": 120},  # Anodic and cathodic slopes
    "HCl": {"ba": 55, "bc": 110},
}

# Experimental electrochemical data from ElectrochemistryResults.csv
# Curry leaf extract in 2M H2SO4 at 25C
CURRY_LEAF_EXPERIMENTAL = {
    # concentration_mg_L: {Ecorr_mV, Icorr_uA_cm2, IE_pct}
    0: {"Ecorr_mV": -546.67, "Icorr_uA_cm2": 7.398, "IE_pct": 0.0},
    1000: {"Ecorr_mV": -692.25, "Icorr_uA_cm2": 11.549, "IE_pct": -56.1},  # Corrosion acceleration
    2000: {"Ecorr_mV": -261.54, "Icorr_uA_cm2": 5.528, "IE_pct": 25.27},
    3000: {"Ecorr_mV": 228.96, "Icorr_uA_cm2": 0.00333, "IE_pct": 99.95},
}

# Aloe vera experimental data from Singh et al. 2016 (1M HCl at 35C)
ALOE_VERA_EXPERIMENTAL = {
    0: {"Ecorr_mV": -469.0, "Icorr_uA_cm2": 731.0, "IE_pct": 0.0, "ba": 73, "bc": 127},
    50: {"Ecorr_mV": -454.0, "Icorr_uA_cm2": 281.0, "IE_pct": 61.5, "ba": 84, "bc": 146},
    100: {"Ecorr_mV": -476.0, "Icorr_uA_cm2": 210.0, "IE_pct": 71.3, "ba": 68, "bc": 130},
    150: {"Ecorr_mV": -479.0, "Icorr_uA_cm2": 149.0, "IE_pct": 79.6, "ba": 92, "bc": 160},
    200: {"Ecorr_mV": -487.0, "Icorr_uA_cm2": 92.0, "IE_pct": 87.4, "ba": 71, "bc": 172},
}


def get_experimental_data(inhibitor_name: str, concentration_mg_L: float) -> dict:
    """
    Get experimental electrochemical data if available.
    Uses linear interpolation for intermediate concentrations.

    Returns:
        dict with Ecorr_mV, Icorr_uA_cm2, IE_pct, ba, bc (or None if not available)
    """
    # Select experimental dataset
    if "Curry leaf" in inhibitor_name:
        exp_data = CURRY_LEAF_EXPERIMENTAL
        default_ba, default_bc = 60, 120  # Typical for H2SO4
    elif "Aloe vera" in inhibitor_name:
        exp_data = ALOE_VERA_EXPERIMENTAL
        default_ba, default_bc = 55, 110  # From literature
    else:
        return None  # No experimental data for this inhibitor

    concentrations = sorted(exp_data.keys())

    # Exact match
    if concentration_mg_L in exp_data:
        result = exp_data[concentration_mg_L].copy()
        if "ba" not in result:
            result["ba"] = default_ba
            result["bc"] = default_bc
        return result

    # Interpolation
    if concentration_mg_L < concentrations[0] or concentration_mg_L > concentrations[-1]:
        return None  # Outside experimental range

    # Find bracketing concentrations
    for i in range(len(concentrations) - 1):
        if concentrations[i] <= concentration_mg_L <= concentrations[i + 1]:
            c1, c2 = concentrations[i], concentrations[i + 1]
            d1, d2 = exp_data[c1], exp_data[c2]

            # Linear interpolation factor
            t = (concentration_mg_L - c1) / (c2 - c1)

            result = {
                "Ecorr_mV": d1["Ecorr_mV"] + t * (d2["Ecorr_mV"] - d1["Ecorr_mV"]),
                "Icorr_uA_cm2": d1["Icorr_uA_cm2"] + t * (d2["Icorr_uA_cm2"] - d1["Icorr_uA_cm2"]),
                "IE_pct": d1["IE_pct"] + t * (d2["IE_pct"] - d1["IE_pct"]),
                "ba": d1.get("ba", default_ba),
                "bc": d1.get("bc", default_bc),
                "interpolated": True,
            }
            return result

    return None

# Feature columns (must match training script)
FEATURE_COLS_NUM = [
    "acid_molarity_M",
    "temperature_C",
    "immersion_time_h",
    "inhibitor_conc_mg_L",
    "log_conc_mg_L",
    "temp_conc_interaction",
    "acid_strength_norm",
    "acid_type_encoded",
]

FEATURE_COLS_CAT = [
    "inhibitor_name",
    "method",
    "acid",
]


def load_models():
    """Load trained ML models and preprocessor."""
    print("\n[INFO] Loading trained models...")

    models = {}

    # Load preprocessor
    preprocessor_path = MODEL_DIR / "best_model_preprocessor.pkl"
    if preprocessor_path.exists():
        models["preprocessor"] = joblib.load(preprocessor_path)
        print(f"   OK: Preprocessor loaded")
    else:
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

    # Load target models
    for target in ["IE", "Ecorr", "Icorr"]:
        model_path = MODEL_DIR / f"best_model_{target}_model.pkl"
        if model_path.exists():
            models[target] = joblib.load(model_path)
            print(f"   OK: {target} model loaded")
        else:
            print(f"   WARN: {target} model not found")

    return models


def create_input_dataframe(
    inhibitor_name: str,
    concentration_mg_L: float,
    acid: str = "H2SO4",
    acid_molarity: float = 1.0,
    temperature_C: float = 30.0,
    immersion_time_h: float = 1.0,
    method: str = "PDP"
) -> pd.DataFrame:
    """Create input DataFrame for model prediction."""

    acid_encoding = {"H2SO4": 0, "HCl": 1, "HNO3": 2}

    data = {
        "inhibitor_name": inhibitor_name,
        "acid": acid,
        "acid_molarity_M": acid_molarity,
        "temperature_C": temperature_C,
        "immersion_time_h": immersion_time_h,
        "inhibitor_conc_mg_L": concentration_mg_L,
        "method": method,
        # Engineered features
        "log_conc_mg_L": np.log10(concentration_mg_L + 1e-3),
        "temp_conc_interaction": temperature_C * concentration_mg_L / 1000.0,
        "acid_strength_norm": acid_molarity / 0.5,
        "acid_type_encoded": acid_encoding.get(acid, 0),
    }

    return pd.DataFrame([data])


def predict_electrochemical_params(
    models: dict,
    inhibitor_name: str,
    concentration_mg_L: float,
    acid: str = "H2SO4",
    acid_molarity: float = 1.0,
    temperature_C: float = 30.0,
    immersion_time_h: float = 1.0,
    method: str = "PDP",
    use_experimental: bool = True
) -> dict:
    """
    Get electrochemical parameters from experimental data or ML models.

    Priority:
    1. Experimental data (if available and use_experimental=True)
    2. ML model predictions (if models trained)
    3. Literature defaults

    Returns:
        dict with Ecorr_mV, Icorr_uA_cm2, IE_pct, ba, bc, source
    """
    results = {"source": "default"}

    # First try experimental data
    if use_experimental:
        exp_data = get_experimental_data(inhibitor_name, concentration_mg_L)
        if exp_data is not None:
            results["Ecorr_mV"] = exp_data["Ecorr_mV"]
            results["Icorr_uA_cm2"] = exp_data["Icorr_uA_cm2"]
            results["IE_pct"] = exp_data["IE_pct"]
            results["ba"] = exp_data.get("ba", 60)
            results["bc"] = exp_data.get("bc", 120)
            results["source"] = "experimental" + (" (interpolated)" if exp_data.get("interpolated") else "")
            return results

    # Fall back to ML predictions
    df_input = create_input_dataframe(
        inhibitor_name, concentration_mg_L, acid,
        acid_molarity, temperature_C, immersion_time_h, method
    )

    X = df_input[FEATURE_COLS_NUM + FEATURE_COLS_CAT]
    preprocessor = models["preprocessor"]
    X_prep = preprocessor.transform(X)

    # Predict each target
    if "Ecorr" in models:
        Ecorr = models["Ecorr"].predict(X_prep)[0]
        results["Ecorr_mV"] = np.clip(Ecorr, -1000, 500)
        results["source"] = "ML model"
    else:
        results["Ecorr_mV"] = -450.0  # Default for mild steel

    if "Icorr" in models:
        Icorr = models["Icorr"].predict(X_prep)[0]
        results["Icorr_uA_cm2"] = np.clip(Icorr, 0.001, 10000)
        results["source"] = "ML model"
    else:
        results["Icorr_uA_cm2"] = 100.0  # Default

    if "IE" in models:
        IE = models["IE"].predict(X_prep)[0]
        results["IE_pct"] = np.clip(IE, -100, 100)  # Allow negative for acceleration
    else:
        results["IE_pct"] = None

    # Default Tafel slopes
    tafel = DEFAULT_TAFEL_SLOPES.get(acid, DEFAULT_TAFEL_SLOPES["H2SO4"])
    results["ba"] = tafel["ba"]
    results["bc"] = tafel["bc"]

    return results


def butler_volmer_current(E, E_corr, i_corr, ba, bc):
    """
    Calculate current density using Butler-Volmer equation.

    Args:
        E: Applied potential (mV vs SCE)
        E_corr: Corrosion potential (mV)
        i_corr: Corrosion current density (uA/cm2)
        ba: Anodic Tafel slope (mV/decade)
        bc: Cathodic Tafel slope (mV/decade)

    Returns:
        i: Current density (uA/cm2), positive for anodic, negative for cathodic
    """
    # Overpotential
    eta = E - E_corr

    # Butler-Volmer equation
    i_anodic = i_corr * (10 ** (eta / ba))  # Anodic component
    i_cathodic = i_corr * (10 ** (-eta / bc))  # Cathodic component

    i_net = i_anodic - i_cathodic

    return i_net


def generate_polarization_curve(
    E_corr: float,
    i_corr: float,
    ba: float = 60,
    bc: float = 120,
    E_range: tuple = (-300, 300),
    n_points: int = 500
) -> tuple:
    """
    Generate polarization curve data.

    Args:
        E_corr: Corrosion potential (mV vs SCE)
        i_corr: Corrosion current density (uA/cm2)
        ba: Anodic Tafel slope (mV/decade)
        bc: Cathodic Tafel slope (mV/decade)
        E_range: Potential range relative to E_corr (mV)
        n_points: Number of data points

    Returns:
        E_values: Array of potential values (mV)
        i_values: Array of current density values (uA/cm2)
    """
    # Generate potential range around Ecorr
    E_values = np.linspace(E_corr + E_range[0], E_corr + E_range[1], n_points)

    # Calculate current for each potential
    i_values = []
    for E in E_values:
        i = butler_volmer_current(E, E_corr, i_corr, ba, bc)
        i_values.append(i)

    return E_values, np.array(i_values)


def plot_single_polarization_curve(
    E_values: np.ndarray,
    i_values: np.ndarray,
    E_corr: float,
    i_corr: float,
    label: str = None,
    output_path: Path = None,
    ax: plt.Axes = None
):
    """Plot a single polarization curve in Tafel format."""

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Convert current to absolute values for log scale
    i_abs = np.abs(i_values)
    i_abs[i_abs < 1e-6] = 1e-6  # Avoid log(0)

    # Plot curve
    ax.semilogy(E_values, i_abs, linewidth=2, label=label)

    # Mark corrosion point
    ax.scatter([E_corr], [i_corr], s=100, c='red', zorder=5,
               label=f'Ecorr={E_corr:.0f}mV, icorr={i_corr:.2f}uA/cm2' if label is None else None)

    ax.set_xlabel("Potential (mV vs SCE)", fontweight='bold', fontsize=12)
    ax.set_ylabel("Current Density (uA/cm2)", fontweight='bold', fontsize=12)
    ax.set_title("Potentiodynamic Polarization Curve", fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best')

    if create_fig:
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"   OK: Saved {output_path}")
        plt.close()

    return ax


def plot_concentration_comparison(
    models: dict,
    inhibitor_name: str,
    concentrations: list = None,
    acid: str = "H2SO4",
    output_path: Path = None
):
    """
    Plot polarization curves for multiple concentrations of the same inhibitor.
    Uses experimental data when available, falls back to ML predictions.
    """
    # Set concentrations based on inhibitor (use experimental data points)
    if concentrations is None:
        if "Curry leaf" in inhibitor_name:
            concentrations = [0, 1000, 2000, 3000]  # Match experimental data
        elif "Aloe vera" in inhibitor_name:
            concentrations = [0, 50, 100, 150, 200]  # Match experimental data
        else:
            concentrations = [0, 100, 300, 500, 1000, 2000]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(concentrations)))

    results_data = []
    data_sources = set()

    for conc, color in zip(concentrations, colors):
        # Get electrochemical parameters (experimental or ML)
        params = predict_electrochemical_params(
            models, inhibitor_name, conc, acid=acid
        )

        E_corr = params["Ecorr_mV"]
        i_corr = params["Icorr_uA_cm2"]
        IE = params.get("IE_pct")
        ba = params.get("ba", 60)
        bc = params.get("bc", 120)
        source = params.get("source", "default")
        data_sources.add(source.split(" ")[0])  # Get main source type

        results_data.append({
            "concentration_mg_L": conc,
            "Ecorr_mV": E_corr,
            "Icorr_uA_cm2": i_corr,
            "IE_pct": IE,
            "source": source
        })

        # Generate and plot curve using actual Tafel slopes
        E_vals, i_vals = generate_polarization_curve(E_corr, i_corr, ba, bc)

        label = f"{conc} mg/L"
        if IE is not None:
            label += f" (IE={IE:.1f}%)"

        # Use different line style for experimental vs predicted
        linestyle = '-' if 'experimental' in source else '--'
        ax1.semilogy(E_vals, np.abs(i_vals) + 1e-6,
                     linewidth=2, color=color, linestyle=linestyle, label=label)
        ax1.scatter([E_corr], [i_corr], s=80, c=[color], edgecolors='white', zorder=5)

    ax1.set_xlabel("Potential (mV vs SCE)", fontweight='bold', fontsize=12)
    ax1.set_ylabel("Current Density (uA/cm2)", fontweight='bold', fontsize=12)

    # Show data source in title
    source_str = ", ".join(sorted(data_sources))
    ax1.set_title(f"Polarization Curves: {inhibitor_name[:40]}\nin {acid}\n(Data: {source_str})",
                  fontweight='bold', fontsize=11)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='best', fontsize=9)

    # Plot Ecorr and Icorr vs concentration
    results_df = pd.DataFrame(results_data)

    ax2_twin = ax2.twinx()

    line1, = ax2.plot(results_df["concentration_mg_L"], results_df["Ecorr_mV"],
                      'b-o', linewidth=2, markersize=8, label='Ecorr (mV)')
    line2, = ax2_twin.plot(results_df["concentration_mg_L"], results_df["Icorr_uA_cm2"],
                           'r-s', linewidth=2, markersize=8, label='Icorr (uA/cm2)')

    ax2.set_xlabel("Inhibitor Concentration (mg/L)", fontweight='bold', fontsize=12)
    ax2.set_ylabel("Ecorr (mV vs SCE)", fontweight='bold', fontsize=12, color='blue')
    ax2_twin.set_ylabel("Icorr (uA/cm2)", fontweight='bold', fontsize=12, color='red')
    ax2.set_title(f"Electrochemical Parameters vs Concentration\n(Data: {source_str})",
                  fontweight='bold', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='red')

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='best')

    # Print results table
    print(f"\n   {inhibitor_name} electrochemical data:")
    print(f"   {'Conc (mg/L)':<12} {'Ecorr (mV)':<12} {'Icorr (uA/cm2)':<15} {'IE (%)':<10} {'Source':<20}")
    print("   " + "-" * 70)
    for _, row in results_df.iterrows():
        ie_str = f"{row['IE_pct']:.1f}" if row['IE_pct'] is not None else "N/A"
        print(f"   {row['concentration_mg_L']:<12.0f} {row['Ecorr_mV']:<12.1f} {row['Icorr_uA_cm2']:<15.4f} {ie_str:<10} {row['source']:<20}")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   OK: Saved {output_path}")

    plt.close()

    return results_df


def plot_inhibitor_comparison(
    models: dict,
    inhibitors: list = None,
    concentration: float = 500,
    acid: str = "H2SO4",
    output_path: Path = None
):
    """
    Plot polarization curves comparing different inhibitors at the same concentration.
    """
    if inhibitors is None:
        inhibitors = ["Curry leaf extract", "Peanut shell extract", "Aloe vera extract"]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Get Tafel slopes
    tafel = DEFAULT_TAFEL_SLOPES.get(acid, DEFAULT_TAFEL_SLOPES["H2SO4"])
    ba, bc = tafel["ba"], tafel["bc"]

    colors = plt.cm.tab10(range(len(inhibitors) + 1))

    # First plot blank (no inhibitor)
    params_blank = predict_electrochemical_params(
        models, inhibitors[0], 0, acid=acid  # 0 concentration = blank
    )
    E_blank, i_blank = generate_polarization_curve(
        params_blank["Ecorr_mV"], params_blank["Icorr_uA_cm2"], ba, bc
    )
    ax.semilogy(E_blank, np.abs(i_blank) + 1e-6,
                linewidth=2, color='black', linestyle='--', label='Blank (no inhibitor)')

    # Plot each inhibitor
    for i, (inhibitor, color) in enumerate(zip(inhibitors, colors[1:])):
        params = predict_electrochemical_params(
            models, inhibitor, concentration, acid=acid
        )

        E_corr = params["Ecorr_mV"]
        i_corr = params["Icorr_uA_cm2"]
        IE = params.get("IE_pct")

        E_vals, i_vals = generate_polarization_curve(E_corr, i_corr, ba, bc)

        label = f"{inhibitor[:30]}"
        if IE is not None:
            label += f" (IE={IE:.1f}%)"

        ax.semilogy(E_vals, np.abs(i_vals) + 1e-6,
                    linewidth=2, color=color, label=label)
        ax.scatter([E_corr], [i_corr], s=100, c=[color], edgecolors='white', zorder=5)

    ax.set_xlabel("Potential (mV vs SCE)", fontweight='bold', fontsize=12)
    ax.set_ylabel("Current Density (uA/cm2)", fontweight='bold', fontsize=12)
    ax.set_title(f"Inhibitor Comparison at {concentration} mg/L in {acid}",
                 fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   OK: Saved {output_path}")

    plt.close()


def interactive_mode(models: dict):
    """Run interactive mode for generating polarization curves."""

    print("\n" + "="*60)
    print("POLARIZATION CURVE GENERATOR - INTERACTIVE MODE")
    print("="*60)

    print("\nAvailable inhibitors:")
    print("  1. Curry leaf extract")
    print("  2. Peanut shell extract")
    print("  3. Aloe vera extract")

    inhibitor_map = {
        "1": "Curry leaf extract",
        "2": "Peanut shell extract",
        "3": "Aloe vera extract",
    }

    while True:
        print("\n" + "-"*40)
        choice = input("\nSelect inhibitor (1-3) or 'q' to quit: ").strip()

        if choice.lower() == 'q':
            break

        inhibitor = inhibitor_map.get(choice)
        if not inhibitor:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue

        try:
            conc = float(input(f"Enter concentration (mg/L) [default=500]: ").strip() or "500")
        except ValueError:
            conc = 500

        acid = input("Enter acid (H2SO4 or HCl) [default=H2SO4]: ").strip().upper() or "H2SO4"
        if acid not in ["H2SO4", "HCL"]:
            acid = "H2SO4"
        if acid == "HCL":
            acid = "HCl"

        # Predict parameters
        print(f"\nPredicting electrochemical parameters for:")
        print(f"  Inhibitor: {inhibitor}")
        print(f"  Concentration: {conc} mg/L")
        print(f"  Acid: {acid}")

        params = predict_electrochemical_params(models, inhibitor, conc, acid=acid)

        print(f"\nPredicted values:")
        print(f"  Ecorr: {params['Ecorr_mV']:.1f} mV vs SCE")
        print(f"  Icorr: {params['Icorr_uA_cm2']:.3f} uA/cm2")
        if params.get("IE_pct"):
            print(f"  IE:    {params['IE_pct']:.1f}%")

        # Generate plot
        output_name = f"polarization_{inhibitor.replace(' ', '_')}_{int(conc)}mgL.png"
        output_path = FIGURES_DIR / output_name

        save = input(f"\nSave plot to {output_path}? (y/n) [default=y]: ").strip().lower()
        if save != 'n':
            plot_concentration_comparison(
                models, inhibitor,
                concentrations=[0, 100, 300, 500, 1000, conc],
                acid=acid,
                output_path=output_path
            )
            print(f"\nPlot saved to: {output_path}")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Generate polarization curves for corrosion inhibitor prediction"
    )
    parser.add_argument("--mode", choices=["interactive", "single", "compare", "all"],
                        default="interactive",
                        help="Operation mode")
    parser.add_argument("--inhibitor", type=str,
                        help="Inhibitor name (for single/compare mode)")
    parser.add_argument("--conc", type=float, default=500,
                        help="Concentration in mg/L (for single mode)")
    parser.add_argument("--acid", type=str, default="H2SO4",
                        help="Acid type (H2SO4 or HCl)")
    parser.add_argument("--output", type=str,
                        help="Output file path")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("POLARIZATION CURVE GENERATOR")
    print("Based on Butler-Volmer equation and ML predictions")
    print("="*60)

    # Load models
    try:
        models = load_models()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please run 02_ml_training_enhanced.py first to train the models.")
        return

    if args.mode == "interactive":
        interactive_mode(models)

    elif args.mode == "single":
        if not args.inhibitor:
            print("\n[ERROR] --inhibitor required for single mode")
            return

        params = predict_electrochemical_params(
            models, args.inhibitor, args.conc, acid=args.acid
        )

        print(f"\nPredicted electrochemical parameters:")
        print(f"  Inhibitor: {args.inhibitor}")
        print(f"  Concentration: {args.conc} mg/L")
        print(f"  Acid: {args.acid}")
        print(f"  Ecorr: {params['Ecorr_mV']:.1f} mV vs SCE")
        print(f"  Icorr: {params['Icorr_uA_cm2']:.3f} uA/cm2")
        if params.get("IE_pct"):
            print(f"  IE: {params['IE_pct']:.1f}%")

        # Generate curve
        output_path = args.output or FIGURES_DIR / "polarization_single.png"
        tafel = DEFAULT_TAFEL_SLOPES.get(args.acid, DEFAULT_TAFEL_SLOPES["H2SO4"])
        E_vals, i_vals = generate_polarization_curve(
            params["Ecorr_mV"], params["Icorr_uA_cm2"],
            tafel["ba"], tafel["bc"]
        )
        plot_single_polarization_curve(
            E_vals, i_vals, params["Ecorr_mV"], params["Icorr_uA_cm2"],
            label=f"{args.inhibitor} @ {args.conc} mg/L",
            output_path=Path(output_path)
        )

    elif args.mode == "compare":
        if not args.inhibitor:
            print("\n[ERROR] --inhibitor required for compare mode")
            return

        output_path = args.output or FIGURES_DIR / f"polarization_compare_{args.inhibitor.replace(' ', '_')}.png"

        plot_concentration_comparison(
            models, args.inhibitor,
            acid=args.acid,
            output_path=Path(output_path)
        )

    elif args.mode == "all":
        print("\nGenerating polarization curves for all inhibitors...")

        inhibitors = ["Curry leaf extract", "Peanut shell extract", "Aloe vera extract"]

        # Generate concentration comparison for each inhibitor
        for inhibitor in inhibitors:
            output_name = f"polarization_{inhibitor.replace(' ', '_')}_comparison.png"
            plot_concentration_comparison(
                models, inhibitor,
                acid=args.acid,
                output_path=FIGURES_DIR / output_name
            )

        # Generate inhibitor comparison
        output_path = args.output or FIGURES_DIR / "polarization_inhibitor_comparison.png"
        plot_inhibitor_comparison(
            models, inhibitors,
            concentration=500,
            acid=args.acid,
            output_path=Path(output_path)
        )

        print(f"\nAll plots saved to: {FIGURES_DIR}")

    print("\n" + "="*60)
    print("DONE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
