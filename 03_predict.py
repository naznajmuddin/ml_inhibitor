#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrosion Inhibitor Prediction Tool
===================================

Use trained ML model to predict inhibition efficiency for new inhibitors
or experimental conditions.

Features:
- Single prediction mode
- Batch prediction from CSV
- Concentration-response curve generation
- Uncertainty quantification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import argparse


def load_model_and_preprocessor(model_dir="ml_models"):
    """Load trained model and preprocessor."""
    print("📂 Loading trained model...")
    
    model_path = Path(model_dir) / "best_model_model.pkl"
    prep_path = Path(model_dir) / "best_model_preprocessor.pkl"
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    
    print("   ✓ Model loaded successfully")
    
    return model, preprocessor


def predict_single(model, preprocessor, conditions):
    """
    Predict IE for a single set of conditions.
    
    Args:
        model: Trained model
        preprocessor: Feature preprocessor
        conditions: Dictionary with experimental conditions
        
    Returns:
        predicted_ie: Predicted inhibition efficiency (%)
    """
    
    # Required features
    required_features = [
        "inhibitor_name", "acid", "steel_grade", "method",
        "acid_molarity_M", "temperature_C", "immersion_time_h",
        "inhibitor_conc_mg_L"
    ]
    
    # Check all required features are present
    for feat in required_features:
        if feat not in conditions:
            raise ValueError(f"Missing required feature: {feat}")
    
    # Add engineered features
    conditions["log_conc_mg_L"] = np.log10(conditions["inhibitor_conc_mg_L"] + 1e-3)
    conditions["temp_conc_interaction"] = (
        conditions["temperature_C"] * conditions["inhibitor_conc_mg_L"] / 1000.0
    )
    conditions["acid_strength_norm"] = conditions["acid_molarity_M"] / 0.5
    
    # Create DataFrame
    df = pd.DataFrame([conditions])
    
    # Define feature columns
    feature_cols = [
        "acid_molarity_M", "temperature_C", "immersion_time_h",
        "inhibitor_conc_mg_L", "log_conc_mg_L", "temp_conc_interaction",
        "acid_strength_norm", "acid", "steel_grade", "method", "inhibitor_name"
    ]
    
    # Preprocess and predict
    X = df[feature_cols]
    X_prep = preprocessor.transform(X)
    prediction = model.predict(X_prep)[0]
    prediction = np.clip(prediction, 0, 100)
    
    return prediction


def predict_batch(model, preprocessor, input_csv, output_csv=None):
    """
    Predict IE for multiple conditions from CSV file.
    
    Args:
        model: Trained model
        preprocessor: Feature preprocessor
        input_csv: Path to input CSV with experimental conditions
        output_csv: Path to save predictions (optional)
        
    Returns:
        df_results: DataFrame with predictions
    """
    print(f"\n📂 Loading batch data from: {input_csv}")
    
    df = pd.read_csv(input_csv)
    print(f"   ✓ Loaded {len(df)} samples")
    
    # Add engineered features
    print("\n🔧 Adding engineered features...")
    df["log_conc_mg_L"] = np.log10(df["inhibitor_conc_mg_L"] + 1e-3)
    df["temp_conc_interaction"] = df["temperature_C"] * df["inhibitor_conc_mg_L"] / 1000.0
    df["acid_strength_norm"] = df["acid_molarity_M"] / 0.5
    
    # Define features
    feature_cols = [
        "acid_molarity_M", "temperature_C", "immersion_time_h",
        "inhibitor_conc_mg_L", "log_conc_mg_L", "temp_conc_interaction",
        "acid_strength_norm", "acid", "steel_grade", "method", "inhibitor_name"
    ]
    
    # Preprocess and predict
    print("\n🔮 Making predictions...")
    X = df[feature_cols]
    X_prep = preprocessor.transform(X)
    predictions = model.predict(X_prep)
    predictions = np.clip(predictions, 0, 100)
    
    df["predicted_IE_pct"] = predictions
    
    # Calculate uncertainty (simple estimate based on test MAE)
    # Replace with actual value from your model metrics
    uncertainty = 15.0  # ±15% typical uncertainty
    df["uncertainty_lower"] = np.clip(predictions - uncertainty, 0, 100)
    df["uncertainty_upper"] = np.clip(predictions + uncertainty, 0, 100)
    
    print(f"   ✓ Predictions complete")
    print(f"\n   Predicted IE range: {predictions.min():.1f}% to {predictions.max():.1f}%")
    
    # Save if output path provided
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Saved predictions to: {output_csv}")
    
    return df


def generate_concentration_response(model, preprocessor, conditions, conc_range=None, 
                                    output_path=None):
    """
    Generate concentration-response curve for a given inhibitor and conditions.
    
    Args:
        model: Trained model
        preprocessor: Feature preprocessor
        conditions: Dict with fixed experimental conditions (without concentration)
        conc_range: Array of concentrations to test (mg/L)
        output_path: Path to save figure (optional)
        
    Returns:
        df_response: DataFrame with concentration vs predicted IE
    """
    print("\n📈 Generating concentration-response curve...")
    
    # Default concentration range if not provided
    if conc_range is None:
        conc_range = np.logspace(0, 3.5, 50)  # 1 to ~3000 mg/L
    
    predictions = []
    
    for conc in conc_range:
        cond = conditions.copy()
        cond["inhibitor_conc_mg_L"] = conc
        
        pred_ie = predict_single(model, preprocessor, cond)
        predictions.append(pred_ie)
    
    predictions = np.array(predictions)
    
    # Create result DataFrame
    df_response = pd.DataFrame({
        "concentration_mg_L": conc_range,
        "predicted_IE_pct": predictions
    })
    
    # Plot if output path provided
    if output_path:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(conc_range, predictions, 'b-', linewidth=2.5, marker='o', 
                markersize=4, markevery=5, label='Predicted IE')
        
        # Add uncertainty band
        uncertainty = 15.0  # ±15%
        ax.fill_between(conc_range, 
                       np.clip(predictions - uncertainty, 0, 100),
                       np.clip(predictions + uncertainty, 0, 100),
                       alpha=0.2, color='blue', label=f'±{uncertainty}% uncertainty')
        
        ax.set_xscale('log')
        ax.set_xlabel('Inhibitor Concentration (mg/L)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Predicted IE (%)', fontweight='bold', fontsize=12)
        ax.set_title(f'Concentration-Response Curve: {conditions.get("inhibitor_name", "Unknown")}',
                    fontweight='bold', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(conc_range.min(), conc_range.max())
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved curve to: {output_path}")
    
    return df_response


def interactive_prediction():
    """Interactive mode for single predictions."""
    
    print("\n" + "="*80)
    print("INTERACTIVE PREDICTION MODE")
    print("="*80)
    
    # Load model
    model, preprocessor = load_model_and_preprocessor()
    
    # Collect inputs
    print("\nEnter experimental conditions:")
    
    conditions = {}
    conditions["inhibitor_name"] = input("  Inhibitor name: ")
    conditions["inhibitor_conc_mg_L"] = float(input("  Concentration (mg/L): "))
    conditions["temperature_C"] = float(input("  Temperature (°C) [default: 25]: ") or "25")
    conditions["acid_molarity_M"] = float(input("  H2SO4 molarity (M) [default: 0.5]: ") or "0.5")
    conditions["immersion_time_h"] = float(input("  Immersion time (hours) [default: 6]: ") or "6")
    
    # Set defaults
    conditions["acid"] = "H2SO4"
    conditions["steel_grade"] = input("  Steel grade [default: ASTM A36]: ") or "ASTM A36"
    conditions["method"] = input("  Method [default: Weight loss]: ") or "Weight loss"
    
    # Predict
    print("\n🔮 Predicting...")
    predicted_ie = predict_single(model, preprocessor, conditions)
    
    # Display result
    print("\n" + "="*80)
    print("PREDICTION RESULT")
    print("="*80)
    print(f"\nInhibitor: {conditions['inhibitor_name']}")
    print(f"Concentration: {conditions['inhibitor_conc_mg_L']:.0f} mg/L")
    print(f"Temperature: {conditions['temperature_C']:.1f}°C")
    print(f"\n>>> Predicted IE: {predicted_ie:.1f}% <<<")
    print(f"\n(Typical uncertainty: ±15%)")
    print("="*80 + "\n")
    
    return predicted_ie


def main():
    """Main function with argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Corrosion Inhibitor IE Prediction Tool"
    )
    
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch", "response_curve"],
        default="interactive",
        help="Prediction mode"
    )
    
    parser.add_argument(
        "--input",
        help="Input CSV file for batch prediction"
    )
    
    parser.add_argument(
        "--output",
        help="Output file path"
    )
    
    parser.add_argument(
        "--inhibitor",
        help="Inhibitor name for response curve"
    )
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        interactive_prediction()
        
    elif args.mode == "batch":
        if not args.input:
            print("Error: --input required for batch mode")
            return
        
        model, preprocessor = load_model_and_preprocessor()
        predict_batch(model, preprocessor, args.input, args.output)
        
    elif args.mode == "response_curve":
        if not args.inhibitor:
            print("Error: --inhibitor required for response_curve mode")
            return
        
        model, preprocessor = load_model_and_preprocessor()
        
        # Standard conditions
        conditions = {
            "inhibitor_name": args.inhibitor,
            "acid": "H2SO4",
            "acid_molarity_M": 0.5,
            "temperature_C": 25.0,
            "immersion_time_h": 6.0,
            "steel_grade": "ASTM A36",
            "method": "Weight loss",
        }
        
        output_path = args.output or f"response_curve_{args.inhibitor.replace(' ', '_')}.png"
        
        df_response = generate_concentration_response(
            model, preprocessor, conditions, output_path=output_path
        )
        
        print(f"\nResponse curve data:")
        print(df_response.to_string(index=False))


if __name__ == "__main__":
    # If no arguments, run interactive mode
    import sys
    if len(sys.argv) == 1:
        interactive_prediction()
    else:
        main()
