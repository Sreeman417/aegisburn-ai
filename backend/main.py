from __future__ import annotations

import inspect
import io
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
STATIC_DIR = BACKEND_DIR / "static"
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"

ANOMALY_MODEL_DIR = MODELS_DIR / "anomaly"
PREDICTION_MODEL_DIR = MODELS_DIR / "prediction"

DEFAULT_DATASET = RAW_DATA_DIR / "component_data.csv"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# =============================================================================
# OPTIONAL PROJECT IMPORTS
# =============================================================================

try:
    from src.anomaly_detection import ParameterModelRegistry
except Exception as error:
    print(f"WARNING: Could not import ParameterModelRegistry: {error}")
    ParameterModelRegistry = None


try:
    from src.drift_prediction import PredictionModelRegistry
except Exception as error:
    print(f"WARNING: Could not import PredictionModelRegistry: {error}")
    PredictionModelRegistry = None


try:
    from src.risk_engine import RiskEngine
except Exception as error:
    print(f"WARNING: Could not import RiskEngine: {error}")
    RiskEngine = None


# =============================================================================
# GLOBAL STATE
# =============================================================================

COMPONENT_DATA: pd.DataFrame | None = None
FEATURE_DATA: pd.DataFrame | None = None

ANOMALY_MODELS: Any = None
PREDICTION_MODELS: Any = None
RISK_ENGINE: Any = None

ACTIVE_DATASET_NAME = DEFAULT_DATASET.name
ACTIVE_DATASET_PATH = DEFAULT_DATASET

ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}


# =============================================================================
# REQUEST MODELS
# =============================================================================

class AnalyzeRequest(BaseModel):
    component_id: str


class UploadCSVRequest(BaseModel):
    filename: str
    csv_text: str


# =============================================================================
# JSON HELPERS
# =============================================================================

def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

    return value


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}

    for key, value in record.items():
        if isinstance(value, dict):
            cleaned[key] = {
                str(k): json_safe(v)
                for k, v in value.items()
            }

        elif isinstance(value, (list, tuple)):
            cleaned[key] = [
                json_safe(item)
                for item in value
            ]

        else:
            cleaned[key] = json_safe(value)

    return cleaned


# =============================================================================
# COLUMN NORMALIZATION
# =============================================================================

def normalize_column_name(column: str) -> str:
    return (
        str(column)
        .strip()
        .lower()
        .replace("\ufeff", "")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def normalize_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [
        normalize_column_name(column)
        for column in df.columns
    ]

    aliases = {
        "componentid": "component_id",
        "component": "component_id",
        "id": "component_id",

        "componenttype": "component_type",
        "type": "component_type",

        "parameter": "parameter_name",
        "parametername": "parameter_name",

        "lot": "lot_id",
        "lotid": "lot_id",

        "temperature": "temperature_c",
        "temperaturec": "temperature_c",

        "value0h": "value_0h",
        "value0": "value_0h",
        "value_0": "value_0h",
        "0h": "value_0h",

        "value24h": "value_24h",
        "value24": "value_24h",
        "value_24": "value_24h",
        "24h": "value_24h",

        "value96h": "value_96h",
        "value96": "value_96h",
        "value_96": "value_96h",
        "96h": "value_96h",

        "value168h": "value_168h",
        "value168": "value_168h",
        "value_168": "value_168h",
        "168h": "value_168h",

        "defect": "defect_type",
        "defecttype": "defect_type",

        "defective": "is_defective",
        "isdefective": "is_defective",
    }

    rename_map: dict[str, str] = {}

    for column in df.columns:
        compact = column.replace("_", "")

        if column in aliases:
            rename_map[column] = aliases[column]

        elif compact in aliases:
            rename_map[column] = aliases[compact]

    df = df.rename(columns=rename_map)

    return df


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_dataset(df: pd.DataFrame) -> tuple[bool, list[str]]:
    required = [
        "component_id",
        "component_type",
        "parameter_name",
        "value_0h",
        "value_24h",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    return len(missing) == 0, missing


# =============================================================================
# DATASET LOADING
# =============================================================================

def is_valid_dataset(path: Path) -> bool:
    try:
        sample = pd.read_csv(path, nrows=5)
        sample = normalize_dataset_columns(sample)

        valid, _ = validate_dataset(sample)
        return valid

    except Exception:
        return False


def find_dataset() -> Path:
    if DEFAULT_DATASET.exists() and is_valid_dataset(DEFAULT_DATASET):
        return DEFAULT_DATASET

    valid_files: list[Path] = []

    for path in BASE_DIR.rglob("*.csv"):
        try:
            relative = path.relative_to(BASE_DIR)
        except Exception:
            relative = path

        parts_lower = {
            part.lower()
            for part in path.parts
        }

        if "models" in parts_lower:
            continue

        if "registry" in path.name.lower():
            continue

        if is_valid_dataset(path):
            valid_files.append(path)

    if not valid_files:
        raise FileNotFoundError(
            "No valid AegisBurn dataset was found."
        )

    valid_files.sort(
        key=lambda p: p.stat().st_size,
        reverse=True,
    )

    return valid_files[0]


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_dataset_columns(df)

    required_columns = [
        "component_id",
        "component_type",
        "parameter_name",
        "value_0h",
        "value_24h",
    ]

    valid, missing = validate_dataset(df)

    if not valid:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    defaults = {
        "lot_id": "UNKNOWN",
        "temperature_c": 25.0,
        "unit": "",
        "value_96h": np.nan,
        "value_168h": np.nan,
    }

    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    string_columns = [
        "component_id",
        "component_type",
        "parameter_name",
        "lot_id",
        "unit",
    ]

    for column in string_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    numeric_columns = [
        "temperature_c",
        "value_0h",
        "value_24h",
        "value_96h",
        "value_168h",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "component_id",
            "component_type",
            "parameter_name",
            "value_0h",
            "value_24h",
        ]
    )

    df = df.reset_index(drop=True)

    return df


def load_dataset_from_path(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return prepare_dataset(df)


def load_dataset() -> pd.DataFrame:
    path = find_dataset()

    print("=" * 70)
    print("AEGISBURN AI DATASET")
    print("=" * 70)
    print(f"Dataset: {path}")

    df = load_dataset_from_path(path)

    global ACTIVE_DATASET_NAME
    global ACTIVE_DATASET_PATH

    ACTIVE_DATASET_NAME = path.name
    ACTIVE_DATASET_PATH = path

    print(f"Rows loaded: {len(df):,}")
    print("=" * 70)

    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()

    features["drift_0_24"] = (
        features["value_24h"]
        - features["value_0h"]
    )

    features["slope_early"] = (
        features["drift_0_24"] / 24.0
    )

    denominator = (
        features["value_0h"]
        .replace(0, np.nan)
    )

    features["ratio_24_0"] = (
        features["value_24h"]
        / denominator
    )

    features["ratio_24_0"] = (
        features["ratio_24_0"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    if "value_96h" in features.columns:
        features["drift_24_96"] = (
            features["value_96h"]
            - features["value_24h"]
        )
    else:
        features["drift_24_96"] = np.nan

    if "value_168h" in features.columns:
        features["drift_96_168"] = (
            features["value_168h"]
            - features["value_96h"]
        )

        features["drift_0_168"] = (
            features["value_168h"]
            - features["value_0h"]
        )
    else:
        features["drift_96_168"] = np.nan
        features["drift_0_168"] = np.nan

    # Group statistics used by anomaly_detection.py.
    grouped = features.groupby(
        [
            "component_type",
            "parameter_name",
        ]
    )

    for column in [
        "value_0h",
        "value_24h",
        "slope_early",
    ]:
        group_mean = grouped[column].transform("mean")
        group_std = grouped[column].transform("std")

        group_std = group_std.replace(
            0,
            np.nan,
        )

        z_score = (
            features[column]
            - group_mean
        ) / group_std

        z_score = (
            z_score
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

        if column == "value_0h":
            features["z_score_0h"] = z_score

        elif column == "value_24h":
            features["z_score_24h"] = z_score

        elif column == "slope_early":
            features["z_score_slope_early"] = z_score

    return features


# =============================================================================
# ENGINEERING LIMITS
# =============================================================================

ENGINEERING_LIMITS = {
    "iddq": 50.0,
    "standby current": 80.0,
    "leakage current": 25.0,
    "propagation delay": 40.0,
}


def get_engineering_limit(
    parameter_name: Any,
) -> float | None:
    if parameter_name is None:
        return None

    name = str(parameter_name).strip().lower()

    if name in ENGINEERING_LIMITS:
        return ENGINEERING_LIMITS[name]

    for parameter, limit in ENGINEERING_LIMITS.items():
        if parameter in name:
            return limit

    return None


# =============================================================================
# MODEL LOADING HELPERS
# =============================================================================

def count_models(registry: Any) -> int:
    if registry is None:
        return 0

    models = getattr(
        registry,
        "models",
        None,
    )

    if isinstance(models, dict):
        return len(models)

    return 0


def try_registry_load(
    registry_class: Any,
    model_dir: Path,
) -> Any:
    if registry_class is None:
        return None

    if not model_dir.exists():
        return None

    load_method = getattr(
        registry_class,
        "load",
        None,
    )

    if load_method is None:
        return None

    attempts = [
        str(model_dir),
        model_dir,
    ]

    for argument in attempts:
        try:
            return load_method(argument)
        except Exception:
            continue

    return None


def find_model_directory(
    preferred: Path,
    alternatives: list[Path],
) -> Path | None:
    candidates = [preferred] + alternatives

    for directory in candidates:
        if directory.exists():
            return directory

    return None


def load_anomaly_models() -> Any:
    directory = find_model_directory(
        ANOMALY_MODEL_DIR,
        [
            MODELS_DIR / "anomaly_models",
            MODELS_DIR / "anomaly_detection",
        ],
    )

    if directory is None:
        print("Anomaly model directory not found.")
        return None

    models = try_registry_load(
        ParameterModelRegistry,
        directory,
    )

    if models is None:
        print("Anomaly models could not be loaded.")
    else:
        print(
            f"Anomaly models loaded: "
            f"{count_models(models)}"
        )

    return models


def load_prediction_models() -> Any:
    directory = find_model_directory(
        PREDICTION_MODEL_DIR,
        [
            MODELS_DIR / "prediction_models",
            MODELS_DIR / "drift_prediction",
            MODELS_DIR / "drift",
        ],
    )

    if directory is None:
        print("Prediction model directory not found.")
        return None

    models = try_registry_load(
        PredictionModelRegistry,
        directory,
    )

    if models is None:
        print("Prediction models could not be loaded.")
    else:
        print(
            f"Prediction models loaded: "
            f"{count_models(models)}"
        )

    return models


# =============================================================================
# RISK ENGINE
# =============================================================================

def initialize_risk_engine() -> Any:
    if RiskEngine is None:
        return None

    try:
        return RiskEngine()
    except Exception as error:
        print(
            f"WARNING: RiskEngine initialization failed: {error}"
        )
        return None


# =============================================================================
# ANOMALY ANALYSIS
# =============================================================================

def heuristic_anomaly(
    row: pd.Series,
) -> dict[str, Any]:
    """
    Prototype fallback only.

    This does NOT use defect_type or is_defective.
    It uses only observed burn-in measurements.
    """

    value_0 = float(row.get("value_0h", 0.0))
    value_24 = float(row.get("value_24h", 0.0))

    drift = value_24 - value_0

    relative_drift = 0.0

    if abs(value_0) > 1e-9:
        relative_drift = abs(drift / value_0)

    z_slope = abs(
        float(
            row.get(
                "z_score_slope_early",
                0.0,
            )
        )
    )

    score = (
        min(relative_drift * 100.0, 100.0) * 0.55
        + min(z_slope * 15.0, 100.0) * 0.45
    )

    score = float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )

    flag = score >= 45.0

    return {
        "anomaly_flag": int(flag),
        "anomaly_label": (
            "ANOMALY"
            if flag
            else "NORMAL"
        ),
        "anomaly_score": score,
        "anomaly_index": score,
        "anomaly_raw_score": score,
    }


def calculate_anomaly(
    row: pd.Series,
) -> dict[str, Any]:

    if ANOMALY_MODELS is None:
        return heuristic_anomaly(row)

    try:
        result = ANOMALY_MODELS.predict_component(row)

        if isinstance(result, dict):
            is_anomaly = result.get(
                "is_anomaly",
                result.get(
                    "anomaly_flag",
                    False,
                ),
            )

            anomaly_score = result.get(
                "anomaly_score",
                result.get(
                    "score",
                    0.0,
                ),
            )

            anomaly_index = result.get(
                "anomaly_index",
                anomaly_score,
            )

            raw_score = result.get(
                "raw_score",
                result.get(
                    "anomaly_raw_score",
                    anomaly_score,
                ),
            )

        else:
            is_anomaly = getattr(
                result,
                "is_anomaly",
                getattr(
                    result,
                    "anomaly_flag",
                    False,
                ),
            )

            anomaly_score = getattr(
                result,
                "anomaly_score",
                getattr(
                    result,
                "score",
                    0.0,
                ),
            )

            anomaly_index = getattr(
                result,
                "anomaly_index",
                anomaly_score,
            )

            raw_score = getattr(
                result,
                "raw_score",
                getattr(
                    result,
                    "anomaly_raw_score",
                    anomaly_score,
                ),
            )

        flag = int(bool(is_anomaly))

        return {
            "anomaly_flag": flag,
            "anomaly_label": (
                "ANOMALY"
                if flag
                else "NORMAL"
            ),
            "anomaly_score": float(anomaly_score),
            "anomaly_index": float(anomaly_index),
            "anomaly_raw_score": float(raw_score),
        }

    except Exception as error:
        print(
            f"Anomaly model failed for "
            f"{row.get('component_id')}: {error}"
        )

        return heuristic_anomaly(row)


# =============================================================================
# PREDICTION
# =============================================================================

def linear_prediction_fallback(
    row: pd.Series,
) -> float:
    """
    Prototype fallback prediction.

    Uses only the observed 0h -> 24h trend.
    """

    value_0 = float(row.get("value_0h", 0.0))
    value_24 = float(row.get("value_24h", value_0))

    slope = (
        value_24 - value_0
    ) / 24.0

    prediction = (
        value_24
        + slope * 144.0
    )

    return float(prediction)


def extract_prediction_value(
    result: Any,
) -> float | None:

    if result is None:
        return None

    if isinstance(result, dict):
        keys = [
            "predicted_168h",
            "prediction",
            "predicted_value",
            "value_168h",
            "predicted_value_168h",
            "future_value",
        ]

        for key in keys:
            if key in result:
                try:
                    value = float(result[key])

                    if np.isfinite(value):
                        return value
                except Exception:
                    pass

    for attribute in [
        "predicted_168h",
        "prediction",
        "predicted_value",
        "value_168h",
        "predicted_value_168h",
        "future_value",
    ]:
        if hasattr(result, attribute):
            try:
                value = float(
                    getattr(
                        result,
                        attribute,
                    )
                )

                if np.isfinite(value):
                    return value
            except Exception:
                pass

    try:
        value = float(result)

        if np.isfinite(value):
            return value
    except Exception:
        pass

    return None


def calculate_prediction(
    row: pd.Series,
) -> float:

    fallback = linear_prediction_fallback(row)

    if PREDICTION_MODELS is None:
        return fallback

    try:
        predict_component = getattr(
            PREDICTION_MODELS,
            "predict_component",
            None,
        )

        if callable(predict_component):
            result = predict_component(row)

            value = extract_prediction_value(result)

            if value is not None:
                return value

        models = getattr(
            PREDICTION_MODELS,
            "models",
            {},
        )

        component_type = str(
            row.get(
                "component_type",
                "",
            )
        ).strip()

        parameter_name = str(
            row.get(
                "parameter_name",
                "",
            )
        ).strip()

        key = (
            f"{component_type}__"
            f"{parameter_name}"
        )

        model = None

        if isinstance(models, dict):
            model = models.get(key)

        if model is not None:
            feature_values = pd.DataFrame(
                [
                    {
                        "value_0h": float(
                            row.get(
                                "value_0h",
                                0.0,
                            )
                        ),
                        "value_24h": float(
                            row.get(
                                "value_24h",
                                0.0,
                            )
                        ),
                    }
                ]
            )

            if hasattr(model, "predict"):
                result = model.predict(
                    feature_values
                )

                value = extract_prediction_value(
                    result[0]
                    if isinstance(
                        result,
                        (list, tuple, np.ndarray)
                    )
                    else result
                )

                if value is not None:
                    return value

        return fallback

    except Exception as error:
        print(
            f"Prediction model failed for "
            f"{row.get('component_id')}: {error}"
        )

        return fallback


# =============================================================================
# RISK FALLBACK
# =============================================================================

def fallback_risk(
    row: pd.Series,
    anomaly_result: dict[str, Any],
    predicted_168h: float,
) -> dict[str, Any]:

    parameter_name = str(
        row.get(
            "parameter_name",
            "",
        )
    )

    value_24 = float(
        row.get(
            "value_24h",
            0.0,
        )
    )

    engineering_limit = get_engineering_limit(
        parameter_name
    )

    if engineering_limit is not None:
        limit_utilization = (
            abs(predicted_168h)
            / engineering_limit
            * 100.0
        )
    else:
        limit_utilization = 0.0

    early_drift = abs(
        float(
            row.get(
                "drift_0_24",
                0.0,
            )
        )
    )

    early_baseline = abs(
        float(
            row.get(
                "value_0h",
                0.0,
            )
        )
    )

    if early_baseline > 1e-9:
        early_drift_percent = (
            early_drift
            / early_baseline
            * 100.0
        )
    else:
        early_drift_percent = 0.0

    if engineering_limit is not None:
        early_limit_percent = (
            abs(value_24)
            / engineering_limit
            * 100.0
        )
    else:
        early_limit_percent = 0.0

    anomaly_component = float(
        anomaly_result.get(
            "anomaly_index",
            0.0,
        )
    )

    future_component = min(
        max(limit_utilization, 0.0),
        100.0,
    )

    drift_component = min(
        max(
            early_drift_percent * 5.0,
            0.0,
        ),
        100.0,
    )

    limit_component = min(
        max(early_limit_percent, 0.0),
        100.0,
    )

    risk_score = (
        anomaly_component * 0.40
        + future_component * 0.35
        + drift_component * 0.15
        + limit_component * 0.10
    )

    risk_score = float(
        np.clip(
            risk_score,
            0.0,
            100.0,
        )
    )

    if risk_score >= 70.0:
        risk_level = "HIGH RISK"

    elif risk_score >= 40.0:
        risk_level = "REVIEW"

    else:
        risk_level = "NORMAL"

    reasons: list[str] = []

    if anomaly_component >= 45:
        reasons.append(
            "Early burn-in behavior shows anomalous characteristics."
        )

    if early_drift_percent >= 5:
        reasons.append(
            "The 0h to 24h parameter drift is elevated."
        )

    if engineering_limit is not None:
        if limit_utilization >= 100:
            reasons.append(
                "Predicted 168h value exceeds the engineering limit."
            )

        elif limit_utilization >= 80:
            reasons.append(
                "Predicted 168h value approaches the engineering limit."
            )

    if not reasons:
        reasons.append(
            "No major early-risk indicator was detected."
        )

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "early_drift_index": float(
            np.clip(
                early_drift_percent,
                0.0,
                100.0,
            )
        ),
        "future_drift_index": float(
            np.clip(
                abs(
                    predicted_168h
                    - value_24
                )
                / max(
                    abs(value_24),
                    1e-9,
                )
                * 100.0,
                0.0,
                100.0,
            )
        ),
        "limit_utilization": float(
            np.clip(
                limit_utilization,
                0.0,
                150.0,
            )
        ),
        "engineering_limit": engineering_limit,
        "reasons": reasons,
    }


# =============================================================================
# RISK ENGINE ADAPTER
# =============================================================================

def calculate_risk(
    row: pd.Series,
    anomaly_result: dict[str, Any],
    predicted_168h: float,
) -> dict[str, Any]:

    # Always prepare the fields expected by the risk engine.
    risk_row = row.copy()

    risk_row["anomaly_flag"] = int(
        anomaly_result.get(
            "anomaly_flag",
            0,
        )
    )

    risk_row["anomaly_score"] = float(
        anomaly_result.get(
            "anomaly_score",
            0.0,
        )
    )

    risk_row["anomaly_index"] = float(
        anomaly_result.get(
            "anomaly_index",
            0.0,
        )
    )

    if RISK_ENGINE is not None:
        try:
            evaluate = getattr(
                RISK_ENGINE,
                "evaluate_component",
                None,
            )

            if callable(evaluate):
                result = evaluate(
                    risk_row,
                    float(predicted_168h),
                    float(
                        abs(
                            row.get(
                                "slope_early",
                                0.0,
                            )
                        )
                    ),
                )

                if isinstance(result, dict):
                    normalized = {
                        "risk_score": float(
                            result.get(
                                "risk_score",
                                0.0,
                            )
                        ),
                        "risk_level": str(
                            result.get(
                                "risk_level",
                                "NORMAL",
                            )
                        ),
                        "early_drift_index": float(
                            result.get(
                                "early_drift_index",
                                0.0,
                            )
                        ),
                        "future_drift_index": float(
                            result.get(
                                "future_drift_index",
                                0.0,
                            )
                        ),
                        "limit_utilization": float(
                            result.get(
                                "limit_utilization",
                                0.0,
                            )
                        ),
                        "engineering_limit": result.get(
                            "engineering_limit",
                            get_engineering_limit(
                                row.get(
                                    "parameter_name"
                                )
                            ),
                        ),
                        "reasons": list(
                            result.get(
                                "reasons",
                                result.get(
                                    "risk_reasons",
                                    [],
                                ),
                            )
                        ),
                    }

                    return normalized

        except Exception as error:
            print(
                f"RiskEngine failed: {error}"
            )
            traceback.print_exc()

    return fallback_risk(
        row,
        anomaly_result,
        predicted_168h,
    )


# =============================================================================
# COMPONENT ANALYSIS
# =============================================================================

def analyze_component(
    component_id: str,
) -> dict[str, Any]:

    component_id = str(component_id)

    if component_id in ANALYSIS_CACHE:
        return ANALYSIS_CACHE[component_id]

    if FEATURE_DATA is None:
        raise RuntimeError(
            "Feature dataset is not loaded."
        )

    matches = FEATURE_DATA[
        FEATURE_DATA["component_id"]
        .astype(str)
        == component_id
    ]

    if matches.empty:
        raise KeyError(
            f"Component not found: {component_id}"
        )

    row = matches.iloc[0].copy()

    # -------------------------------------------------------------------------
    # 1. ANOMALY
    # -------------------------------------------------------------------------

    anomaly_result = calculate_anomaly(row)

    row["anomaly_flag"] = int(
        anomaly_result["anomaly_flag"]
    )

    row["anomaly_score"] = float(
        anomaly_result["anomaly_score"]
    )

    row["anomaly_index"] = float(
        anomaly_result["anomaly_index"]
    )

    row["anomaly_raw_score"] = float(
        anomaly_result["anomaly_raw_score"]
    )

    # -------------------------------------------------------------------------
    # 2. FUTURE PREDICTION
    # -------------------------------------------------------------------------

    predicted_168h = calculate_prediction(row)

    value_24h = float(
        row.get(
            "value_24h",
            0.0,
        )
    )

    predicted_slope = (
        predicted_168h
        - value_24h
    ) / 144.0

    # -------------------------------------------------------------------------
    # 3. RISK
    # -------------------------------------------------------------------------

    risk_result = calculate_risk(
        row,
        anomaly_result,
        predicted_168h,
    )

    engineering_limit = risk_result.get(
        "engineering_limit"
    )

    if engineering_limit is None:
        engineering_limit = get_engineering_limit(
            row.get("parameter_name")
        )

    limit_utilization = float(
        risk_result.get(
            "limit_utilization",
            0.0,
        )
    )

    # -------------------------------------------------------------------------
    # 4. REASONS
    # -------------------------------------------------------------------------

    reasons = risk_result.get(
        "reasons",
        [],
    )

    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    # -------------------------------------------------------------------------
    # 5. FINAL RESULT
    # -------------------------------------------------------------------------

    result = {
        **row.to_dict(),

        **anomaly_result,

        "predicted_168h": float(
            predicted_168h
        ),

        "predicted_slope": float(
            predicted_slope
        ),

        "engineering_limit": (
            engineering_limit
        ),

        "limit_utilization": (
            limit_utilization
        ),

        "risk_score": float(
            risk_result.get(
                "risk_score",
                0.0,
            )
        ),

        "risk_level": str(
            risk_result.get(
                "risk_level",
                "NORMAL",
            )
        ),

        "early_drift_index": float(
            risk_result.get(
                "early_drift_index",
                0.0,
            )
        ),

        "future_drift_index": float(
            risk_result.get(
                "future_drift_index",
                0.0,
            )
        ),

        "risk_reasons": reasons,

        "reasons": reasons,
    }

    # Frontend-friendly aliases.
    result["risk_decision"] = result["risk_level"]

    result["early_drift"] = float(
        row.get(
            "drift_0_24",
            0.0,
        )
    )

    result["future_drift"] = float(
        predicted_168h
        - value_24h
    )

    result["prediction_unit"] = str(
        row.get(
            "unit",
            "",
        )
    )

    result["parameter_unit"] = str(
        row.get(
            "unit",
            "",
        )
    )

    result = clean_record(result)

    ANALYSIS_CACHE[component_id] = result

    return result


# =============================================================================
# DATASET REPLACEMENT
# =============================================================================

def replace_dataset(
    df: pd.DataFrame,
    filename: str,
) -> None:

    global COMPONENT_DATA
    global FEATURE_DATA
    global ACTIVE_DATASET_NAME
    global ACTIVE_DATASET_PATH
    global ANALYSIS_CACHE

    prepared = prepare_dataset(df)

    if prepared.empty:
        raise ValueError(
            "Uploaded CSV contains no valid component rows."
        )

    COMPONENT_DATA = prepared
    FEATURE_DATA = create_features(
        COMPONENT_DATA
    )

    ACTIVE_DATASET_NAME = filename
    ACTIVE_DATASET_PATH = Path(filename)

    ANALYSIS_CACHE = {}

    print("=" * 70)
    print("ACTIVE DATASET UPDATED")
    print("=" * 70)
    print(f"Name: {filename}")
    print(f"Rows: {len(prepared):,}")
    print("=" * 70)


# =============================================================================
# APPLICATION LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global COMPONENT_DATA
    global FEATURE_DATA
    global ANOMALY_MODELS
    global PREDICTION_MODELS
    global RISK_ENGINE
    global ANALYSIS_CACHE

    print("=" * 70)
    print("STARTING AEGISBURN AI")
    print("=" * 70)

    try:
        COMPONENT_DATA = load_dataset()

        FEATURE_DATA = create_features(
            COMPONENT_DATA
        )

        print(
            f"Feature rows: "
            f"{len(FEATURE_DATA):,}"
        )

    except Exception as error:
        print(
            f"ERROR loading dataset: {error}"
        )
        traceback.print_exc()

        COMPONENT_DATA = pd.DataFrame()
        FEATURE_DATA = pd.DataFrame()

    ANOMALY_MODELS = load_anomaly_models()
    PREDICTION_MODELS = load_prediction_models()

    RISK_ENGINE = initialize_risk_engine()

    ANALYSIS_CACHE = {}

    print("=" * 70)
    print("AEGISBURN AI — BACKEND READY")
    print("=" * 70)

    yield

    print("AEGISBURN AI shutting down.")


# =============================================================================
# FASTAPI
# =============================================================================

app = FastAPI(
    title="AegisBurn AI",
    version="1.0.0",
    description=(
        "AI-driven burn-in anomaly detection, "
        "drift prediction and predictive component screening."
    ),
    lifespan=lifespan,
)


# =============================================================================
# CORS
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# STATIC FILES
# =============================================================================

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(
            directory=str(STATIC_DIR)
        ),
        name="static",
    )


# =============================================================================
# DASHBOARD
# =============================================================================

@app.get(
    "/",
    include_in_schema=False,
)
async def dashboard():

    index_file = (
        STATIC_DIR
        / "index.html"
    )

    if not index_file.exists():
        return JSONResponse(
            status_code=404,
            content={
                "detail": (
                    "Dashboard not found. "
                    "Expected backend/static/index.html"
                )
            },
        )

    return FileResponse(
        str(index_file)
    )


# =============================================================================
# HEALTH
# =============================================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "system": "AegisBurn AI",
        "anomaly_models": count_models(
            ANOMALY_MODELS
        ),
        "prediction_models": count_models(
            PREDICTION_MODELS
        ),
        "risk_engine": (
            RISK_ENGINE is not None
        ),
        "dataset_loaded": (
            COMPONENT_DATA is not None
            and not COMPONENT_DATA.empty
        ),
    }


# =============================================================================
# METADATA
# =============================================================================

@app.get("/metadata")
async def metadata():

    total = 0

    if COMPONENT_DATA is not None:
        total = len(COMPONENT_DATA)

    component_types: list[str] = []
    parameters: list[str] = []

    if COMPONENT_DATA is not None and not COMPONENT_DATA.empty:
        component_types = sorted(
            COMPONENT_DATA[
                "component_type"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        parameters = sorted(
            COMPONENT_DATA[
                "parameter_name"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    return {
        "system": "AegisBurn AI",
        "version": "1.0.0",
        "status": "running",
        "total": total,
        "rows": total,
        "dataset_name": ACTIVE_DATASET_NAME,
        "dataset_path": str(
            ACTIVE_DATASET_PATH
        ),
        "component_types": component_types,
        "parameters": parameters,
        "anomaly_models": count_models(
            ANOMALY_MODELS
        ),
        "prediction_models": count_models(
            PREDICTION_MODELS
        ),
        "risk_engine": (
            RISK_ENGINE is not None
        ),
        "message": (
            "AI-driven anomaly detection "
            "and burn-in drift prediction."
        ),
    }


# =============================================================================
# COMPONENT LIST
# =============================================================================

@app.get("/components")
async def get_components():

    if COMPONENT_DATA is None or COMPONENT_DATA.empty:
        raise HTTPException(
            status_code=503,
            detail="Dataset is not loaded.",
        )

    columns = [
        "component_id",
        "component_type",
        "parameter_name",
        "unit",
        "lot_id",
        "temperature_c",
        "value_0h",
        "value_24h",
        "value_96h",
        "value_168h",
    ]

    available_columns = [
        column
        for column in columns
        if column in COMPONENT_DATA.columns
    ]

    records = COMPONENT_DATA[
        available_columns
    ].to_dict(
        orient="records"
    )

    return [
        clean_record(record)
        for record in records
    ]


# =============================================================================
# SINGLE COMPONENT
# =============================================================================

@app.get(
    "/components/{component_id}"
)
async def get_component(
    component_id: str,
):

    if COMPONENT_DATA is None:
        raise HTTPException(
            status_code=503,
            detail="Dataset is not loaded.",
        )

    matches = COMPONENT_DATA[
        COMPONENT_DATA["component_id"]
        .astype(str)
        == str(component_id)
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Component not found: "
                f"{component_id}"
            ),
        )

    return clean_record(
        matches.iloc[0].to_dict()
    )


# =============================================================================
# ANALYZE — GET
# =============================================================================

@app.get(
    "/analyze/{component_id}"
)
async def analyze_component_get(
    component_id: str,
):

    try:
        return analyze_component(
            component_id
        )

    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# =============================================================================
# ANALYZE — POST
# =============================================================================

@app.post("/analyze")
async def analyze_component_post(
    request: AnalyzeRequest,
):

    try:
        return analyze_component(
            request.component_id
        )

    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# =============================================================================
# CSV UPLOAD
# =============================================================================

@app.post("/upload-csv")
async def upload_csv(
    request: UploadCSVRequest,
):

    global ANALYSIS_CACHE

    filename = (
        request.filename
        .strip()
        or "uploaded_dataset.csv"
    )

    if not filename.lower().endswith(".csv"):
        filename += ".csv"

    if not request.csv_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV is empty.",
        )

    try:
        csv_bytes = request.csv_text.encode(
            "utf-8-sig"
        )

        df = pd.read_csv(
            io.BytesIO(csv_bytes)
        )

        prepared = prepare_dataset(df)

        if prepared.empty:
            raise ValueError(
                "No valid rows were found."
            )

        replace_dataset(
            prepared,
            filename,
        )

        return {
            "status": "success",
            "message": (
                "CSV uploaded successfully."
            ),
            "filename": filename,
            "rows": len(prepared),
            "dataset_name": filename,
        }

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not process CSV: "
                f"{error}"
            ),
        )


# =============================================================================
# REFRESH
# =============================================================================

@app.post("/refresh")
async def refresh():

    global ANALYSIS_CACHE

    ANALYSIS_CACHE = {}

    return {
        "status": "success",
        "message": "Analysis cache cleared.",
    }


# =============================================================================
# RUN APPLICATION DIRECTLY
# =============================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )