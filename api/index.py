from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from scipy import stats

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "ml_models"
MODEL_PATH = MODEL_DIR / "best_model_IE_model.pkl"
PREP_PATH = MODEL_DIR / "best_model_preprocessor.pkl"
DATASET_PATH = BASE_DIR / "corrosion_inhibitors_expanded_v2.csv"
FIGURE_DIRS = {
    "ml_figures": BASE_DIR / "ml_figures",
    "study_figures": BASE_DIR / "study_figures",
}
INHIBITOR_COLORS = {
    "Curry leaf extract": "#2ecc71",
    "Peanut shell extract": "#e74c3c",
    "Aloe vera extract": "#3498db",
}

REQUIRED_FIELDS = [
    "inhibitor_name",
    "method",
    "acid",
    "acid_molarity_M",
    "temperature_C",
    "immersion_time_h",
    "inhibitor_conc_mg_L",
]

FEATURE_COLS = [
    "acid_molarity_M",
    "temperature_C",
    "immersion_time_h",
    "inhibitor_conc_mg_L",
    "log_conc_mg_L",
    "temp_conc_interaction",
    "acid_strength_norm",
    "acid_type_encoded",
    "acid",
    "inhibitor_name",
    "method",
]

_model = None
_preprocessor = None


def load_model_and_preprocessor():
    global _model, _preprocessor

    if _model is None or _preprocessor is None:
        _model = joblib.load(MODEL_PATH)
        _preprocessor = joblib.load(PREP_PATH)

    return _model, _preprocessor


def format_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    return value


def df_to_rows(df: pd.DataFrame) -> list[list]:
    return [[format_value(value) for value in row] for row in df.values.tolist()]


def build_graph_tables():
    if not DATASET_PATH.exists():
        return [
            {
                "title": "Study graphs data",
                "columns": [],
                "rows": [],
                "error": "Dataset not found.",
            }
        ]

    df = pd.read_csv(DATASET_PATH)
    tables = []

    # 01. Concentration vs IE (individual)
    conc_rows = []
    for inhibitor in INHIBITOR_COLORS.keys():
        data = df[df["inhibitor_name"] == inhibitor]
        if data.empty:
            continue
        grouped = (
            data.groupby("inhibitor_conc_mg_L")["inhibition_efficiency_pct"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grouped.insert(0, "inhibitor_name", inhibitor)
        conc_rows.append(grouped)
    if conc_rows:
        conc_table = pd.concat(conc_rows, ignore_index=True)
        tables.append(
            {
                "title": "01 Concentration vs IE (grouped)",
                "columns": conc_table.columns.tolist(),
                "rows": df_to_rows(conc_table),
            }
        )

    # 02. Concentration comparison (all inhibitors)
    if conc_rows:
        tables.append(
            {
                "title": "02 Concentration comparison (grouped)",
                "columns": conc_table.columns.tolist(),
                "rows": df_to_rows(conc_table),
            }
        )

    # 03. Concentration log scale
    log_rows = []
    for inhibitor in INHIBITOR_COLORS.keys():
        data = df[
            (df["inhibitor_name"] == inhibitor) & (df["inhibitor_conc_mg_L"] > 0)
        ]
        if data.empty:
            continue
        grouped = (
            data.groupby("inhibitor_conc_mg_L")["inhibition_efficiency_pct"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grouped["log10_conc_mg_L"] = np.log10(grouped["inhibitor_conc_mg_L"])
        grouped.insert(0, "inhibitor_name", inhibitor)
        log_rows.append(grouped)
    if log_rows:
        log_table = pd.concat(log_rows, ignore_index=True)
        tables.append(
            {
                "title": "03 Concentration log-scale data",
                "columns": log_table.columns.tolist(),
                "rows": df_to_rows(log_table),
            }
        )

    # 04. Acid comparison
    acid_summary_rows = []
    for inhibitor in INHIBITOR_COLORS.keys():
        for acid in ["H2SO4", "HCl"]:
            data = df[
                (df["inhibitor_name"] == inhibitor)
                & (df["acid"] == acid)
                & (df["inhibitor_conc_mg_L"] > 0)
            ]
            if data.empty:
                mean_ie = std_ie = count = 0
            else:
                mean_ie = data["inhibition_efficiency_pct"].mean()
                std_ie = data["inhibition_efficiency_pct"].std()
                count = len(data)
            acid_summary_rows.append(
                {
                    "inhibitor_name": inhibitor,
                    "acid": acid,
                    "mean_ie": mean_ie,
                    "std_ie": std_ie,
                    "count": count,
                }
            )
    acid_summary = pd.DataFrame(acid_summary_rows)
    tables.append(
        {
            "title": "04 Acid comparison (bar summary)",
            "columns": acid_summary.columns.tolist(),
            "rows": df_to_rows(acid_summary),
        }
    )

    acid_scatter = df[
        (df["acid"].isin(["H2SO4", "HCl"])) & (df["inhibitor_conc_mg_L"] > 0)
    ][
        [
            "inhibitor_name",
            "acid",
            "inhibitor_conc_mg_L",
            "inhibition_efficiency_pct",
        ]
    ].copy()
    if not acid_scatter.empty:
        tables.append(
            {
                "title": "04 Acid comparison (scatter points)",
                "columns": acid_scatter.columns.tolist(),
                "rows": df_to_rows(acid_scatter),
            }
        )

    # 05. Temperature effect
    temp_rows = []
    for inhibitor in INHIBITOR_COLORS.keys():
        data = df[
            (df["inhibitor_name"] == inhibitor) & (df["inhibitor_conc_mg_L"] > 0)
        ]
        if data.empty:
            continue
        grouped = (
            data.groupby("temperature_C")["inhibition_efficiency_pct"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grouped.insert(0, "inhibitor_name", inhibitor)
        temp_rows.append(grouped)
    if temp_rows:
        temp_table = pd.concat(temp_rows, ignore_index=True)
        tables.append(
            {
                "title": "05 Temperature effect (mean/std by temperature)",
                "columns": temp_table.columns.tolist(),
                "rows": df_to_rows(temp_table),
            }
        )

    aloe_data = df[
        (df["inhibitor_name"] == "Aloe vera extract")
        & (df["inhibitor_conc_mg_L"] > 0)
    ]
    if not aloe_data.empty:
        pivot = aloe_data.pivot_table(
            values="inhibition_efficiency_pct",
            index="temperature_C",
            columns="inhibitor_conc_mg_L",
            aggfunc="mean",
        )
        if pivot.size:
            pivot_flat = pivot.reset_index().melt(
                id_vars="temperature_C",
                var_name="inhibitor_conc_mg_L",
                value_name="mean_ie",
            )
            tables.append(
                {
                    "title": "05 Aloe vera heatmap values",
                    "columns": pivot_flat.columns.tolist(),
                    "rows": df_to_rows(pivot_flat),
                }
            )

    # 06. Adsorption isotherm
    iso_rows = []
    fit_rows = []
    theta_rows = []
    for inhibitor in INHIBITOR_COLORS.keys():
        data = df[
            (df["inhibitor_name"] == inhibitor)
            & (df["inhibitor_conc_mg_L"] > 0)
            & (df["inhibition_efficiency_pct"] > 0)
            & (df["inhibition_efficiency_pct"] < 100)
        ]
        if len(data) >= 3:
            C = data["inhibitor_conc_mg_L"].values
            theta = data["inhibition_efficiency_pct"].values / 100
            C_over_theta = C / theta
            for conc, tval, c_over in zip(C, theta, C_over_theta):
                iso_rows.append(
                    {
                        "inhibitor_name": inhibitor,
                        "inhibitor_conc_mg_L": conc,
                        "theta": tval,
                        "C_over_theta": c_over,
                    }
                )
            slope, intercept, r_value, _, _ = stats.linregress(C, C_over_theta)
            fit_rows.append(
                {
                    "inhibitor_name": inhibitor,
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_value**2,
                }
            )

        data_theta = df[
            (df["inhibitor_name"] == inhibitor)
            & (df["inhibitor_conc_mg_L"] > 0)
            & (df["inhibition_efficiency_pct"] > 0)
        ]
        if len(data_theta) >= 3:
            grouped = data_theta.groupby("inhibitor_conc_mg_L")[
                "inhibition_efficiency_pct"
            ].mean()
            for conc, mean_ie in grouped.items():
                theta_rows.append(
                    {
                        "inhibitor_name": inhibitor,
                        "inhibitor_conc_mg_L": conc,
                        "mean_ie": mean_ie,
                        "theta": mean_ie / 100,
                    }
                )
    if iso_rows:
        iso_table = pd.DataFrame(iso_rows)
        tables.append(
            {
                "title": "06 Langmuir linearized data (C/theta vs C)",
                "columns": iso_table.columns.tolist(),
                "rows": df_to_rows(iso_table),
            }
        )
    if fit_rows:
        fit_table = pd.DataFrame(fit_rows)
        tables.append(
            {
                "title": "06 Langmuir fit summary",
                "columns": fit_table.columns.tolist(),
                "rows": df_to_rows(fit_table),
            }
        )
    if theta_rows:
        theta_table = pd.DataFrame(theta_rows)
        tables.append(
            {
                "title": "06 Surface coverage data (theta vs concentration)",
                "columns": theta_table.columns.tolist(),
                "rows": df_to_rows(theta_table),
            }
        )

    # 07. Electrochemical data
    if "Ecorr_mV" in df.columns:
        echem_data = df[df["Ecorr_mV"].notna()].copy()
        if not echem_data.empty:
            echem_table = echem_data[
                [
                    "inhibitor_name",
                    "inhibitor_conc_mg_L",
                    "Ecorr_mV",
                    "Icorr_uA_cm2",
                    "inhibition_efficiency_pct",
                ]
            ]
            tables.append(
                {
                    "title": "07 Electrochemical data points",
                    "columns": echem_table.columns.tolist(),
                    "rows": df_to_rows(echem_table),
                }
            )

    # 08. Max IE comparison
    max_rows = []
    for inhibitor in INHIBITOR_COLORS.keys():
        data = df[df["inhibitor_name"] == inhibitor]
        if data.empty:
            max_rows.append(
                {
                    "inhibitor_name": inhibitor,
                    "max_ie": 0,
                    "optimal_conc_mg_L": 0,
                }
            )
            continue
        max_idx = data["inhibition_efficiency_pct"].idxmax()
        max_rows.append(
            {
                "inhibitor_name": inhibitor,
                "max_ie": data.loc[max_idx, "inhibition_efficiency_pct"],
                "optimal_conc_mg_L": data.loc[max_idx, "inhibitor_conc_mg_L"],
            }
        )
    max_table = pd.DataFrame(max_rows)
    tables.append(
        {
            "title": "08 Maximum IE and optimal concentration",
            "columns": max_table.columns.tolist(),
            "rows": df_to_rows(max_table),
        }
    )

    # 09. IE distribution data
    dist_table = df[df["inhibitor_conc_mg_L"] > 0][
        ["inhibitor_name", "inhibition_efficiency_pct"]
    ].copy()
    if not dist_table.empty:
        tables.append(
            {
                "title": "09 IE distribution values",
                "columns": dist_table.columns.tolist(),
                "rows": df_to_rows(dist_table),
            }
        )

    # 10. Model predictions (train/val/test)
    train_path = BASE_DIR / "preprocessed_data/train_data.csv"
    val_path = BASE_DIR / "preprocessed_data/val_data.csv"
    test_path = BASE_DIR / "preprocessed_data/test_data.csv"
    if train_path.exists() and val_path.exists() and test_path.exists():
        feature_cols = [
            "acid_molarity_M",
            "temperature_C",
            "immersion_time_h",
            "inhibitor_conc_mg_L",
            "log_conc_mg_L",
            "temp_conc_interaction",
            "acid_strength_norm",
            "acid_type_encoded",
            "acid",
            "inhibitor_name",
            "method",
        ]
        try:
            model, preprocessor = load_model_and_preprocessor()
            split_tables = []
            for name, path in [
                ("Train", train_path),
                ("Validation", val_path),
                ("Test", test_path),
            ]:
                data = pd.read_csv(path)
                if data.empty:
                    continue
                X = data[feature_cols]
                y_true = data["inhibition_efficiency_pct"].values
                y_pred = model.predict(preprocessor.transform(X))
                y_pred = np.clip(y_pred, 0, 100)
                split_df = pd.DataFrame(
                    {
                        "dataset_split": name,
                        "inhibitor_name": data["inhibitor_name"],
                        "actual_ie": y_true,
                        "predicted_ie": y_pred,
                    }
                )
                split_tables.append(split_df)
            if split_tables:
                pred_table = pd.concat(split_tables, ignore_index=True)
                tables.append(
                    {
                        "title": "10 Model predictions (actual vs predicted)",
                        "columns": pred_table.columns.tolist(),
                        "rows": df_to_rows(pred_table),
                    }
                )
            else:
                tables.append(
                    {
                        "title": "10 Model predictions (actual vs predicted)",
                        "columns": [],
                        "rows": [],
                        "error": "No rows found in preprocessed datasets.",
                    }
                )
        except Exception as exc:
            tables.append(
                {
                    "title": "10 Model predictions (actual vs predicted)",
                    "columns": [],
                    "rows": [],
                    "error": f"Prediction data unavailable: {exc}",
                }
            )
    else:
        tables.append(
            {
                "title": "10 Model predictions (actual vs predicted)",
                "columns": [],
                "rows": [],
                "error": "Preprocessed train/val/test data not found.",
            }
        )

    # 11. Feature importance
    fi_path = BASE_DIR / "ml_models/best_model_IE_feature_importance.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path).sort_values("importance", ascending=True).tail(15)
        tables.append(
            {
                "title": "11 Feature importance (top 15)",
                "columns": fi_df.columns.tolist(),
                "rows": df_to_rows(fi_df),
            }
        )
    else:
        tables.append(
            {
                "title": "11 Feature importance (top 15)",
                "columns": [],
                "rows": [],
                "error": "Feature importance file not found.",
            }
        )

    # 12. Correlation matrix
    num_cols = [
        "acid_molarity_M",
        "temperature_C",
        "immersion_time_h",
        "inhibitor_conc_mg_L",
        "inhibition_efficiency_pct",
        "Ecorr_mV",
        "Icorr_uA_cm2",
    ]
    num_cols = [col for col in num_cols if col in df.columns]
    if num_cols:
        corr = df[num_cols].corr()
        corr_df = corr.reset_index().rename(columns={"index": "variable"})
        tables.append(
            {
                "title": "12 Correlation matrix values",
                "columns": corr_df.columns.tolist(),
                "rows": df_to_rows(corr_df),
            }
        )

    # 13. Summary infographic data
    summary_rows = [
        {"metric": "total_experiments", "value": len(df)},
        {
            "metric": "ie_range",
            "value": f"{df['inhibition_efficiency_pct'].min():.1f} to {df['inhibition_efficiency_pct'].max():.1f}",
        },
        {"metric": "mean_ie", "value": df["inhibition_efficiency_pct"].mean()},
        {
            "metric": "concentration_range",
            "value": f"{df['inhibitor_conc_mg_L'].min():.0f} to {df['inhibitor_conc_mg_L'].max():.0f}",
        },
        {
            "metric": "temperature_range",
            "value": f"{df['temperature_C'].min():.0f} to {df['temperature_C'].max():.0f}",
        },
    ]
    summary_table = pd.DataFrame(summary_rows)
    tables.append(
        {
            "title": "13 Summary infographic: dataset stats",
            "columns": summary_table.columns.tolist(),
            "rows": df_to_rows(summary_table),
        }
    )

    metrics_path = BASE_DIR / "ml_models/best_model_IE_metrics.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        tables.append(
            {
                "title": "13 Summary infographic: model metrics",
                "columns": metrics_df.columns.tolist(),
                "rows": df_to_rows(metrics_df),
            }
        )
    else:
        tables.append(
            {
                "title": "13 Summary infographic: model metrics",
                "columns": [],
                "rows": [],
                "error": "Model metrics file not found.",
            }
        )

    # 14. 3D surface grid data + actual points
    from scipy.interpolate import griddata

    surface_rows = []
    surface_points = []
    for inhibitor in INHIBITOR_COLORS.keys():
        data = df[
            (df["inhibitor_name"] == inhibitor) & (df["inhibitor_conc_mg_L"] > 0)
        ]
        if len(data) < 3:
            continue
        conc_range = np.linspace(
            data["inhibitor_conc_mg_L"].min(),
            data["inhibitor_conc_mg_L"].max(),
            20,
        )
        temp_range = np.linspace(
            data["temperature_C"].min(), data["temperature_C"].max(), 20
        )
        C, T = np.meshgrid(conc_range, temp_range)
        points = data[["inhibitor_conc_mg_L", "temperature_C"]].values
        values = data["inhibition_efficiency_pct"].values
        IE = griddata(points, values, (C, T), method="linear")

        for conc, temp, ie in zip(C.flatten(), T.flatten(), IE.flatten()):
            surface_rows.append(
                {
                    "inhibitor_name": inhibitor,
                    "inhibitor_conc_mg_L": conc,
                    "temperature_C": temp,
                    "interpolated_ie": ie,
                }
            )

        for _, row in data.iterrows():
            surface_points.append(
                {
                    "inhibitor_name": inhibitor,
                    "inhibitor_conc_mg_L": row["inhibitor_conc_mg_L"],
                    "temperature_C": row["temperature_C"],
                    "observed_ie": row["inhibition_efficiency_pct"],
                }
            )

    if surface_rows:
        surface_table = pd.DataFrame(surface_rows)
        tables.append(
            {
                "title": "14 3D surface interpolated grid",
                "columns": surface_table.columns.tolist(),
                "rows": df_to_rows(surface_table),
            }
        )

    if surface_points:
        surface_points_table = pd.DataFrame(surface_points)
        tables.append(
            {
                "title": "14 3D surface observed points",
                "columns": surface_points_table.columns.tolist(),
                "rows": df_to_rows(surface_points_table),
            }
        )

    return tables


def normalize_conditions(conditions: dict) -> dict:
    for field in REQUIRED_FIELDS:
        if field not in conditions:
            raise ValueError(f"Missing required field: {field}")

    normalized = dict(conditions)
    normalized["inhibitor_name"] = str(normalized["inhibitor_name"])
    normalized["method"] = str(normalized["method"])
    normalized["acid"] = str(normalized["acid"])
    normalized["acid_molarity_M"] = float(normalized["acid_molarity_M"])
    normalized["temperature_C"] = float(normalized["temperature_C"])
    normalized["immersion_time_h"] = float(normalized["immersion_time_h"])
    normalized["inhibitor_conc_mg_L"] = float(normalized["inhibitor_conc_mg_L"])

    normalized["log_conc_mg_L"] = np.log10(
        normalized["inhibitor_conc_mg_L"] + 1e-3
    )
    normalized["temp_conc_interaction"] = (
        normalized["temperature_C"] * normalized["inhibitor_conc_mg_L"] / 1000.0
    )
    normalized["acid_strength_norm"] = normalized["acid_molarity_M"] / 0.5
    normalized["acid_type_encoded"] = 1 if normalized["acid"] == "HCl" else 0

    return normalized


def predict_records(records: list[dict]) -> list[dict]:
    model, preprocessor = load_model_and_preprocessor()
    normalized_records = [normalize_conditions(record) for record in records]

    df = pd.DataFrame(normalized_records)
    X = df[FEATURE_COLS]
    X_prep = preprocessor.transform(X)
    predictions = model.predict(X_prep)
    predictions = np.clip(predictions, 0, 100)

    uncertainty = 5.0
    results = []
    for record, prediction in zip(normalized_records, predictions):
        results.append(
            {
                "input": record,
                "predicted_IE_pct": float(prediction),
                "uncertainty_lower": float(max(prediction - uncertainty, 0)),
                "uncertainty_upper": float(min(prediction + uncertainty, 100)),
            }
        )

    return results


@app.get("/")
def index():
    return render_template(
        "index.html",
        result=None,
        error=None,
        defaults={
            "inhibitor_name": "Curry leaf extract",
            "method": "Weight loss",
            "acid": "H2SO4",
            "acid_molarity_M": 1.0,
            "temperature_C": 25.0,
            "immersion_time_h": 1.0,
            "inhibitor_conc_mg_L": 500.0,
        },
    )


@app.get("/api")
def api_info():
    return jsonify(
        {
            "message": "Corrosion inhibitor prediction API",
            "endpoints": {
                "GET /health": "health check",
                "POST /predict": "single prediction (JSON)",
                "POST /predict/batch": "batch prediction (JSON)",
                "GET /api/samples": "sample rows from dataset (JSON)",
                "GET /data": "dataset table (UI)",
                "GET /graphs": "generated graphs (UI)",
            },
        }
    )


@app.get("/data")
def data_table():
    if not DATASET_PATH.exists():
        return render_template(
            "data.html", columns=[], rows=[], error="Dataset not found."
        )

    df = pd.read_csv(DATASET_PATH)
    columns = df.columns.tolist()
    rows = df.fillna("").values.tolist()

    return render_template(
        "data.html",
        columns=columns,
        rows=rows,
        error=None,
        row_count=len(rows),
    )


@app.get("/graphs")
def graphs():
    polarization_files = []
    study_files = []

    ml_dir = FIGURE_DIRS["ml_figures"]
    if ml_dir.exists():
        polarization_files = sorted(
            [path for path in ml_dir.iterdir() if path.suffix == ".png"]
        )

    study_dir = FIGURE_DIRS["study_figures"]
    if study_dir.exists():
        study_files = sorted(
            [path for path in study_dir.iterdir() if path.suffix == ".png"]
        )

    def build_items(paths, folder):
        items = []
        for path in paths:
            title = path.stem.replace("_", " ")
            items.append(
                {
                    "title": title,
                    "url": url_for("media", folder=folder, filename=path.name),
                }
            )
        return items

    polarization_items = [
        item
        for item in build_items(polarization_files, "ml_figures")
        if item["title"].startswith("polarization")
    ]
    study_items = build_items(study_files, "study_figures")

    return render_template(
        "graphs.html",
        polarization_items=polarization_items,
        study_items=study_items,
        graph_tables=build_graph_tables(),
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Expected JSON object payload"}), 400

    try:
        result = predict_records([payload])[0]
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


@app.post("/predict/ui")
def predict_ui():
    form_data = request.form.to_dict(flat=True)
    error = None
    result = None

    try:
        result = predict_records([form_data])[0]
    except Exception as exc:
        error = str(exc)

    return render_template(
        "index.html",
        result=result,
        error=error,
        defaults=form_data,
    )


@app.post("/predict/batch")
def predict_batch():
    payload = request.get_json(silent=True)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("items")
    else:
        records = None

    if not records:
        return (
            jsonify(
                {"error": "Expected a JSON list or an object with an 'items' list"}
            ),
            400,
        )

    try:
        results = predict_records(records)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"count": len(results), "results": results})


@app.get("/api/samples")
def dataset_samples():
    if not DATASET_PATH.exists():
        return jsonify({"error": "Dataset not found"}), 404

    df = pd.read_csv(DATASET_PATH)
    sample_cols = [
        "inhibitor_name",
        "method",
        "acid",
        "acid_molarity_M",
        "temperature_C",
        "immersion_time_h",
        "inhibitor_conc_mg_L",
    ]
    missing_cols = [col for col in sample_cols if col not in df.columns]
    if missing_cols:
        return jsonify({"error": f"Dataset missing columns: {missing_cols}"}), 400

    samples_df = df[sample_cols].dropna().head(50).copy()

    def normalize_method(value: str) -> str:
        lower = value.strip().lower()
        if "polarization" in lower:
            return "PDP"
        if "impedance" in lower:
            return "EIS"
        if "weight" in lower:
            return "Weight loss"
        return value

    samples_df["method"] = samples_df["method"].astype(str).map(normalize_method)
    samples = samples_df.to_dict(orient="records")
    return jsonify({"count": len(samples), "samples": samples})


@app.get("/media/<folder>/<path:filename>")
def media(folder, filename):
    directory = FIGURE_DIRS.get(folder)
    if directory is None:
        return jsonify({"error": "Invalid folder"}), 404
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    app.run(debug=True)
