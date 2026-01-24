#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Literature Data to Training Dataset
========================================

This script helps merge new electrochemical data from literature
into the main training dataset.

Usage:
    python 05_add_literature_data.py --input new_data.csv
    python 05_add_literature_data.py --input electrochemical_data_template.csv --preview
"""

import pandas as pd
import argparse
from pathlib import Path


MAIN_DATASET = "corrosion_inhibitors_expanded_v2.csv"
REQUIRED_COLS = [
    "paper_id", "inhibitor_name", "steel_grade", "acid",
    "acid_molarity_M", "temperature_C", "inhibitor_conc_mg_L",
    "method", "inhibition_efficiency_pct"
]
ELECTROCHEMICAL_COLS = ["Ecorr_mV", "Icorr_uA_cm2", "ba_mV_dec", "bc_mV_dec"]


def load_new_data(filepath: str) -> pd.DataFrame:
    """Load new data from CSV, skipping comment lines."""

    # Read file and filter out comment lines
    with open(filepath, 'r') as f:
        lines = [line for line in f if not line.strip().startswith('#')]

    # Parse as CSV
    from io import StringIO
    df = pd.read_csv(StringIO(''.join(lines)))

    # Remove any rows that look like headers or examples
    df = df[~df['paper_id'].str.contains('Example', na=False)]

    return df


def validate_data(df: pd.DataFrame) -> tuple:
    """Validate the new data and return (valid_df, issues)."""

    issues = []

    # Check required columns
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        return None, issues

    # Check for missing required values
    for col in REQUIRED_COLS:
        null_count = df[col].isna().sum()
        if null_count > 0:
            issues.append(f"Column '{col}' has {null_count} missing values")

    # Check value ranges
    if 'inhibition_efficiency_pct' in df.columns:
        ie_out_of_range = df[
            (df['inhibition_efficiency_pct'] < -100) |
            (df['inhibition_efficiency_pct'] > 100)
        ]
        if len(ie_out_of_range) > 0:
            issues.append(f"IE% values outside [-100, 100] range: {len(ie_out_of_range)} rows")

    if 'temperature_C' in df.columns:
        temp_invalid = df[df['temperature_C'] < 0]
        if len(temp_invalid) > 0:
            issues.append(f"Negative temperature values: {len(temp_invalid)} rows")

    if 'inhibitor_conc_mg_L' in df.columns:
        conc_invalid = df[df['inhibitor_conc_mg_L'] < 0]
        if len(conc_invalid) > 0:
            issues.append(f"Negative concentration values: {len(conc_invalid)} rows")

    # Check electrochemical data
    echem_count = df[ELECTROCHEMICAL_COLS].notna().sum()
    print(f"\n[INFO] Electrochemical data availability:")
    for col, count in echem_count.items():
        print(f"   {col}: {count}/{len(df)} rows")

    return df, issues


def merge_datasets(main_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new data into main dataset, avoiding duplicates."""

    # Identify potential duplicates based on key columns
    key_cols = ["paper_id", "inhibitor_name", "inhibitor_conc_mg_L", "method", "temperature_C"]

    # Create merge keys
    main_df['_merge_key'] = main_df[key_cols].astype(str).agg('_'.join, axis=1)
    new_df['_merge_key'] = new_df[key_cols].astype(str).agg('_'.join, axis=1)

    # Find duplicates
    duplicates = new_df[new_df['_merge_key'].isin(main_df['_merge_key'])]
    new_unique = new_df[~new_df['_merge_key'].isin(main_df['_merge_key'])]

    if len(duplicates) > 0:
        print(f"\n[WARN] Found {len(duplicates)} potential duplicate rows (skipped):")
        for _, row in duplicates.iterrows():
            print(f"   - {row['paper_id']}: {row['inhibitor_name']} @ {row['inhibitor_conc_mg_L']} mg/L")

    # Remove merge key column
    main_df = main_df.drop('_merge_key', axis=1)
    new_unique = new_unique.drop('_merge_key', axis=1)

    # Merge
    merged_df = pd.concat([main_df, new_unique], ignore_index=True)

    return merged_df, len(new_unique)


def print_summary(df: pd.DataFrame):
    """Print summary of the dataset."""

    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)

    print(f"\nTotal rows: {len(df)}")

    print("\nInhibitors:")
    for inh in df['inhibitor_name'].unique():
        count = len(df[df['inhibitor_name'] == inh])
        ecorr_count = df[df['inhibitor_name'] == inh]['Ecorr_mV'].notna().sum()
        print(f"   {inh}: {count} rows ({ecorr_count} with Ecorr/Icorr)")

    print("\nAcids:")
    for acid in df['acid'].unique():
        count = len(df[df['acid'] == acid])
        print(f"   {acid}: {count} rows")

    print("\nMethods:")
    for method in df['method'].unique():
        count = len(df[df['method'] == method])
        print(f"   {method}: {count} rows")

    print("\nElectrochemical data:")
    print(f"   Rows with Ecorr: {df['Ecorr_mV'].notna().sum()}")
    print(f"   Rows with Icorr: {df['Icorr_uA_cm2'].notna().sum()}")
    print(f"   Rows with Tafel slopes: {df['ba_mV_dec'].notna().sum()}")


def main():
    parser = argparse.ArgumentParser(
        description="Add literature data to training dataset"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to new data CSV file")
    parser.add_argument("--preview", "-p", action="store_true",
                        help="Preview changes without saving")
    parser.add_argument("--output", "-o", default=MAIN_DATASET,
                        help=f"Output file (default: {MAIN_DATASET})")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("ADD LITERATURE DATA TO TRAINING DATASET")
    print("="*60)

    # Load main dataset
    print(f"\n[INFO] Loading main dataset: {MAIN_DATASET}")
    main_df = pd.read_csv(MAIN_DATASET)
    print(f"   Current rows: {len(main_df)}")

    # Load new data
    print(f"\n[INFO] Loading new data: {args.input}")
    try:
        new_df = load_new_data(args.input)
        print(f"   New rows: {len(new_df)}")
    except Exception as e:
        print(f"[ERROR] Failed to load new data: {e}")
        return

    if len(new_df) == 0:
        print("\n[INFO] No new data rows found (only comments/examples in file)")
        return

    # Validate
    print("\n[INFO] Validating new data...")
    validated_df, issues = validate_data(new_df)

    if issues:
        print("\n[WARN] Validation issues found:")
        for issue in issues:
            print(f"   - {issue}")

    if validated_df is None:
        print("\n[ERROR] Cannot proceed due to validation errors")
        return

    # Merge
    print("\n[INFO] Merging datasets...")
    merged_df, added_count = merge_datasets(main_df, validated_df)

    print(f"\n[INFO] Added {added_count} new rows")
    print(f"   Total rows: {len(merged_df)}")

    # Summary
    print_summary(merged_df)

    # Save or preview
    if args.preview:
        print("\n[PREVIEW MODE] No changes saved")
        print(f"   Would save {len(merged_df)} rows to {args.output}")
    else:
        # Backup original
        backup_path = Path(MAIN_DATASET).with_suffix('.csv.bak')
        main_df.to_csv(backup_path, index=False)
        print(f"\n[INFO] Backup saved: {backup_path}")

        # Save merged
        merged_df.to_csv(args.output, index=False)
        print(f"[INFO] Updated dataset saved: {args.output}")

        print("\n[NEXT STEPS]")
        print("   1. Run: python 01_data_preprocessing.py")
        print("   2. Run: python 02_ml_training_enhanced.py")
        print("   3. Run: python 04_polarization_curves.py --mode all")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
