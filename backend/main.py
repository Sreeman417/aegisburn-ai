from __future__ import annotations

import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

BACKEND_DIR = BASE_DIR / "backend"
STATIC_DIR = BACKEND_DIR / "static"

MODELS_DIR = BASE_DIR / "models"
ANOMALY_MODEL_DIR = MODELS_DIR / "anomaly"
PREDICTION_MODEL_DIR = MODELS_DIR / "prediction"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

try:
    from src.anomaly_detection import ParameterModelRegistry

except Exception as error:

    print(
        "WARNING: Could not import "
        f"ParameterModelRegistry: {error}"
    )

    ParameterModelRegistry = None


try:
    from src.drift_prediction import PredictionModelRegistry

except Exception as error:

    print(
        "WARNING: Could not import "
        f"PredictionModelRegistry: {error}"
    )

    PredictionModelRegistry = None


try:
    from src.risk_engine import RiskEngine

except Exception as error:

    print(
        f"WARNING: Could not import RiskEngine: {error}"
    )

    RiskEngine = None


# =============================================================================
# GLOBAL STATE
# =============================================================================

COMPONENT_DATA: pd.DataFrame | None = None
FEATURE_DATA: pd.DataFrame | None = None

ANOMALY_MODELS: Any = None
PREDICTION_MODELS: Any = None
RISK_ENGINE: Any = None

ANALYSIS_CACHE: dict[str, dict] = {}


# =============================================================================
# REQUEST MODEL
# =============================================================================

class AnalyzeRequest(BaseModel):

    component_id: str


# =============================================================================
# JSON UTILITIES
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

    return value


def clean_record(record: dict) -> dict:

    cleaned = {}

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
# DATASET HELPERS
# =============================================================================

def normalize_column_name(column: str) -> str:

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def is_valid_dataset(path: Path) -> bool:

    try:

        sample = pd.read_csv(
            path,
            nrows=5,
        )

    except Exception:

        return False

    columns = [

        normalize_column_name(column)

        for column in sample.columns
    ]

    compact_columns = {

        column.replace("_", "")

        for column in columns
    }

    has_component_id = (

        "component_id" in columns

        or "componentid" in compact_columns
    )

    has_component_type = (

        "component_type" in columns

        or "componenttype" in compact_columns
    )

    has_parameter = (

        "parameter_name" in columns

        or "parametername" in compact_columns

        or "parameter" in columns
    )

    has_value_0h = (

        "value_0h" in columns

        or "value0h" in compact_columns

        or "0h" in columns
    )

    has_value_24h = (

        "value_24h" in columns

        or "value24h" in compact_columns

        or "24h" in columns
    )

    return (

        has_component_id

        and has_component_type

        and has_parameter

        and has_value_0h

        and has_value_24h
    )


# =============================================================================
# FIND DATASET
# =============================================================================

def find_dataset() -> Path:

    print("=" * 70)
    print("SEARCHING FOR AEGISBURN COMPONENT DATASET")
    print("=" * 70)

    all_csv_files = list(
        BASE_DIR.rglob("*.csv")
    )

    valid_files = []

    for path in all_csv_files:

        try:

            relative_path = path.relative_to(
                BASE_DIR
            )

        except Exception:

            relative_path = path

        parts_lower = [

            part.lower()

            for part in path.parts
        ]

        if "models" in parts_lower:

            print(
                f"Skipping model CSV: "
                f"{relative_path}"
            )

            continue

        filename_lower = path.name.lower()

        if "registry" in filename_lower:

            print(
                f"Skipping registry CSV: "
                f"{relative_path}"
            )

            continue

        print(
            f"Checking CSV: "
            f"{relative_path}"
        )

        if is_valid_dataset(path):

            print(
                f"VALID DATASET FOUND: "
                f"{relative_path}"
            )

            valid_files.append(path)

    if not valid_files:

        raise FileNotFoundError(
            "NO VALID AEGISBURN DATASET FOUND."
        )

    valid_files.sort(
        key=lambda path: path.stat().st_size,
        reverse=True,
    )

    selected_path = valid_files[0]

    print("=" * 70)
    print("DATASET SELECTED")
    print("=" * 70)

    print(
        f"Dataset: {selected_path}"
    )

    print("=" * 70)

    return selected_path


# =============================================================================
# LOAD DATASET
# =============================================================================

def load_dataset() -> pd.DataFrame:

    dataset_path = find_dataset()

    print(
        f"Loading dataset: {dataset_path}"
    )

    df = pd.read_csv(
        dataset_path
    )

    print(
        f"Raw rows: {len(df):,}"
    )

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
        "value_0": "value_0h",
        "0h": "value_0h",

        "value24h": "value_24h",
        "value_24": "value_24h",
        "24h": "value_24h",

        "value96h": "value_96h",
        "value_96": "value_96h",
        "96h": "value_96h",

        "value168h": "value_168h",
        "value_168": "value_168h",
        "168h": "value_168h",
    }

    rename_map = {}

    for column in df.columns:

        compact = column.replace(
            "_",
            "",
        )

        if column in aliases:

            rename_map[column] = aliases[column]

        elif compact in aliases:

            rename_map[compact] = aliases[compact]

    df = df.rename(
        columns=rename_map
    )

    required_columns = [

        "component_id",
        "component_type",
        "parameter_name",
        "value_0h",
        "value_24h",
    ]

    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Dataset is missing required columns: "
            f"{missing_columns}"
        )

    if "lot_id" not in df.columns:
        df["lot_id"] = "UNKNOWN"

    if "temperature_c" not in df.columns:
        df["temperature_c"] = 25.0

    if "unit" not in df.columns:
        df["unit"] = ""

    if "value_96h" not in df.columns:
        df["value_96h"] = np.nan

    if "value_168h" not in df.columns:
        df["value_168h"] = np.nan

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

    before_count = len(df)

    df = df.dropna(

        subset=[

            "component_id",
            "component_type",
            "parameter_name",
            "value_0h",
            "value_24h",
        ]
    )

    df = df.reset_index(
        drop=True
    )

    removed_count = (
        before_count - len(df)
    )

    print("=" * 70)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Components loaded: {len(df):,}"
    )

    print(
        f"Invalid rows removed: "
        f"{removed_count:,}"
    )

    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    features = df.copy()

    features["drift_0_24"] = (

        features["value_24h"]

        - features["value_0h"]
    )

    features["slope_early"] = (

        features["drift_0_24"]

        / 24.0
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

    # Required by your anomaly_detection.py
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

        group_mean = grouped[
            column
        ].transform("mean")

        group_std = grouped[
            column
        ].transform("std")

        group_std = group_std.replace(
            0,
            np.nan,
        )

        z_score = (
            features[column] - group_mean
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

            features[
                "z_score_slope_early"
            ] = z_score

    return features


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_anomaly_models():

    if ParameterModelRegistry is None:

        return None

    if not ANOMALY_MODEL_DIR.exists():

        print(
            f"Anomaly model directory missing: "
            f"{ANOMALY_MODEL_DIR}"
        )

        return None

    try:

        return ParameterModelRegistry.load(
            str(ANOMALY_MODEL_DIR)
        )

    except Exception as error:

        print(
            f"Could not load anomaly models: "
            f"{error}"
        )

        traceback.print_exc()

        return None


def load_prediction_models():

    if PredictionModelRegistry is None:

        return None

    if not PREDICTION_MODEL_DIR.exists():

        print(
            f"Prediction model directory missing: "
            f"{PREDICTION_MODEL_DIR}"
        )

        return None

    try:

        return PredictionModelRegistry.load(
            str(PREDICTION_MODEL_DIR)
        )

    except Exception as error:

        print(
            f"Could not load prediction models: "
            f"{error}"
        )

        return None


# =============================================================================
# RISK ENGINE
# =============================================================================

def initialize_risk_engine():

    if RiskEngine is None:

        return None

    try:

        return RiskEngine()

    except Exception as error:

        print(
            f"Could not initialize RiskEngine: "
            f"{error}"
        )

        return None


# =============================================================================
# ANOMALY ANALYSIS
# =============================================================================

def calculate_anomaly(
    row: pd.Series
) -> dict:

    default = {

        "anomaly_flag": 0,
        "anomaly_label": "NORMAL",
        "anomaly_score": 0.0,
        "anomaly_index": 0.0,
        "anomaly_raw_score": 0.0,
    }

    if ANOMALY_MODELS is None:

        return default

    try:

        result = (
            ANOMALY_MODELS
            .predict_component(row)
        )

        # Your anomaly module returns AnomalyResult:
        #
        # is_anomaly
        # anomaly_score
        # raw_score
        #
        # It does NOT return anomaly_flag directly.

        if isinstance(result, dict):

            flag = result.get(
                "is_anomaly",
                result.get(
                    "anomaly_flag",
                    result.get(
                        "flag",
                        0,
                    )
                )
            )

            score = result.get(
                "anomaly_score",
                result.get(
                    "score",
                    0.0,
                )
            )

            index = result.get(
                "anomaly_index",
                score,
            )

            raw_score = result.get(
                "raw_score",
                result.get(
                    "anomaly_raw_score",
                    score,
                )
            )

        else:

            flag = getattr(
                result,
                "is_anomaly",
                getattr(
                    result,
                    "anomaly_flag",
                    getattr(
                        result,
                        "flag",
                        0,
                    )
                )
            )

            score = getattr(
                result,
                "anomaly_score",
                getattr(
                    result,
                    "score",
                    0.0,
                )
            )

            index = getattr(
                result,
                "anomaly_index",
                score,
            )

            raw_score = getattr(
                result,
                "raw_score",
                getattr(
                    result,
                    "anomaly_raw_score",
                    score,
                )
            )

        flag = int(
            bool(flag)
        )

        return {

            "anomaly_flag": flag,

            "anomaly_label": (

                "ANOMALY"

                if flag

                else "NORMAL"
            ),

            "anomaly_score": float(score),

            "anomaly_index": float(index),

            "anomaly_raw_score": float(raw_score),
        }

    except Exception as error:

        print(
            f"Anomaly analysis failed for "
            f"{row.get('component_id')}: "
            f"{error}"
        )

        traceback.print_exc()

        return default


# =============================================================================
# PREDICTION
# =============================================================================

def calculate_prediction(
    row: pd.Series
) -> float:

    fallback = row.get(
        "value_168h",
        row.get(
            "value_24h",
            0.0,
        )
    )

    fallback = pd.to_numeric(
        fallback,
        errors="coerce",
    )

    if pd.isna(fallback):

        fallback = pd.to_numeric(
            row.get(
                "value_24h",
                0.0,
            ),
            errors="coerce",
        )

    if pd.isna(fallback):

        fallback = 0.0

    fallback = float(fallback)

    if PREDICTION_MODELS is None:

        return fallback

    try:

        result = (
            PREDICTION_MODELS
            .predict_component(row)
        )

        if isinstance(result, dict):

            for key in [

                "predicted_168h",
                "prediction",
                "predicted_value",
            ]:

                if key in result:

                    value = float(
                        result[key]
                    )

                    if np.isfinite(value):

                        return value

        if hasattr(
            result,
            "predicted_168h",
        ):

            value = float(
                result.predicted_168h
            )

            if np.isfinite(value):

                return value

        value = float(result)

        if np.isfinite(value):

            return value

        return fallback

    except Exception as error:

        print(
            f"Prediction failed for "
            f"{row.get('component_id')}: "
            f"{error}"
        )

        return fallback


# =============================================================================
# SAFETY SLOPE
# =============================================================================

def calculate_safety_slope(
    row: pd.Series
) -> float:

    slope = abs(
        float(
            row.get(
                "slope_early",
                0.0,
            )
        )
    )

    return max(
        slope,
        0.001,
    )


# =============================================================================
# RISK ANALYSIS
# =============================================================================

def calculate_risk(
    row: pd.Series,
    anomaly_result: dict,
    predicted_168h: float,
    safety_slope: float,
) -> dict:

    default = {

        "risk_score": 0.0,
        "risk_level": "NORMAL",
        "early_drift_index": 0.0,
        "future_drift_index": 0.0,
        "limit_utilization": 0.0,
        "engineering_limit": None,
        "reasons": [],
    }

    if RISK_ENGINE is None:

        return default

    try:

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

        result = (
            RISK_ENGINE
            .evaluate_component(
                risk_row,
                float(predicted_168h),
                float(safety_slope),
            )
        )

        if isinstance(result, dict):

            return {

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

                "engineering_limit": (
                    result.get(
                        "engineering_limit"
                    )
                ),

                "reasons": result.get(
                    "reasons",
                    [],
                ),
            }

        return {

            "risk_score": float(
                getattr(
                    result,
                    "risk_score",
                    0.0,
                )
            ),

            "risk_level": str(
                getattr(
                    result,
                    "risk_level",
                    "NORMAL",
                )
            ),

            "early_drift_index": float(
                getattr(
                    result,
                    "early_drift_index",
                    0.0,
                )
            ),

            "future_drift_index": float(
                getattr(
                    result,
                    "future_drift_index",
                    0.0,
                )
            ),

            "limit_utilization": float(
                getattr(
                    result,
                    "limit_utilization",
                    0.0,
                )
            ),

            "engineering_limit": getattr(
                result,
                "engineering_limit",
                None,
            ),

            "reasons": list(
                getattr(
                    result,
                    "reasons",
                    [],
                )
            ),
        }

    except Exception as error:

        print(
            f"Risk calculation failed: {error}"
        )

        traceback.print_exc()

        return default


# =============================================================================
# COMPLETE COMPONENT ANALYSIS
# =============================================================================

def analyze_component(
    component_id: str
) -> dict:

    if component_id in ANALYSIS_CACHE:

        return ANALYSIS_CACHE[
            component_id
        ]

    if FEATURE_DATA is None:

        raise RuntimeError(
            "Feature dataset is not loaded."
        )

    matches = FEATURE_DATA[

        FEATURE_DATA[
            "component_id"
        ]
        .astype(str)

        == str(component_id)
    ]

    if matches.empty:

        raise KeyError(
            f"Component not found: "
            f"{component_id}"
        )

    row = matches.iloc[0].copy()

    # -------------------------------------------------------------
    # 1. ANOMALY
    # -------------------------------------------------------------

    anomaly_result = calculate_anomaly(
        row
    )

    row["anomaly_flag"] = int(
        anomaly_result.get(
            "anomaly_flag",
            0,
        )
    )

    row["anomaly_score"] = float(
        anomaly_result.get(
            "anomaly_score",
            0.0,
        )
    )

    row["anomaly_index"] = float(
        anomaly_result.get(
            "anomaly_index",
            0.0,
        )
    )

    row["anomaly_raw_score"] = float(
        anomaly_result.get(
            "anomaly_raw_score",
            0.0,
        )
    )

    # -------------------------------------------------------------
    # 2. PREDICTION
    # -------------------------------------------------------------

    predicted_168h = calculate_prediction(
        row
    )

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

    # -------------------------------------------------------------
    # 3. SAFETY SLOPE
    # -------------------------------------------------------------

    safety_slope = calculate_safety_slope(
        row
    )

    # -------------------------------------------------------------
    # 4. RISK
    # -------------------------------------------------------------

    risk_result = calculate_risk(
        row,
        anomaly_result,
        predicted_168h,
        safety_slope,
    )

    # -------------------------------------------------------------
    # 5. COMPLETE RESULT
    # -------------------------------------------------------------

    result = {

        **row.to_dict(),

        **anomaly_result,

        "predicted_168h":
            predicted_168h,

        "predicted_slope":
            predicted_slope,

        "safety_slope":
            safety_slope,

        **risk_result,
    }

    result = clean_record(
        result
    )

    ANALYSIS_CACHE[
        component_id
    ] = result

    return result


# =============================================================================
# APPLICATION LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI
):

    global COMPONENT_DATA
    global FEATURE_DATA
    global ANOMALY_MODELS
    global PREDICTION_MODELS
    global RISK_ENGINE
    global ANALYSIS_CACHE
    print("=" * 70)

    print("Loading dataset...")

    COMPONENT_DATA = load_dataset()

    print(
        "Creating early features..."
    )

    FEATURE_DATA = create_features(
        COMPONENT_DATA
    )

    print(
        "Early features created."
    )

    print(
        "Loading anomaly models..."
    )

    ANOMALY_MODELS = (
        load_anomaly_models()
    )

    anomaly_count = 0

    if (

        ANOMALY_MODELS is not None

        and hasattr(
            ANOMALY_MODELS,
            "models",
        )
    ):

        anomaly_count = len(
            ANOMALY_MODELS.models
        )

    print(
        f"Anomaly models loaded: "
        f"{anomaly_count}"
    )

    print(
        "Loading prediction models..."
    )

    PREDICTION_MODELS = (
        load_prediction_models()
    )

    prediction_count = 0

    if (

        PREDICTION_MODELS is not None

        and hasattr(
            PREDICTION_MODELS,
            "models",
        )
    ):

        prediction_count = len(
            PREDICTION_MODELS.models
        )

    print(
        f"Prediction models loaded: "
        f"{prediction_count}"
    )

    print(
        "Initializing risk engine..."
    )

    RISK_ENGINE = (
        initialize_risk_engine()
    )

    ANALYSIS_CACHE = {}

    print(
        "On-demand analysis enabled."
    )

    print("=" * 70)
    print("AEGISBURN AI — BACKEND READY")
    print("=" * 70)

    yield

    print(
        "AEGISBURN AI shutting down."
    )


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(

    title="AegisBurn AI",

    version="1.0.0",

    description=(
        "AI-driven anomaly detection "
        "and burn-in drift prediction."
    ),

    lifespan=lifespan,
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

    anomaly_count = 0
    prediction_count = 0

    if (

        ANOMALY_MODELS is not None

        and hasattr(
            ANOMALY_MODELS,
            "models",
        )
    ):

        anomaly_count = len(
            ANOMALY_MODELS.models
        )

    if (

        PREDICTION_MODELS is not None

        and hasattr(
            PREDICTION_MODELS,
            "models",
        )
    ):

        prediction_count = len(
            PREDICTION_MODELS.models
        )

    return {

        "status": "healthy",

        "system": "AegisBurn AI",

        "anomaly_models":
            anomaly_count,

        "prediction_models":
            prediction_count,
    }


# =============================================================================
# METADATA
# =============================================================================

@app.get("/metadata")
async def metadata():

    total = 0

    if COMPONENT_DATA is not None:

        total = len(
            COMPONENT_DATA
        )

    return {

        "system": "AegisBurn AI",

        "version": "1.0.0",

        "status": "running",

        "total": total,

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

    if COMPONENT_DATA is None:

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
        "defect_type",
        "is_defective",
    ]

    available_columns = [

        column

        for column in columns

        if column in COMPONENT_DATA.columns
    ]

    records = (

        COMPONENT_DATA[
            available_columns
        ]

        .to_dict(
            orient="records"
        )
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
    component_id: str
):

    if COMPONENT_DATA is None:

        raise HTTPException(
            status_code=503,
            detail="Dataset is not loaded.",
        )

    matches = COMPONENT_DATA[

        COMPONENT_DATA[
            "component_id"
        ]
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
# ANALYZE COMPONENT - GET
# =============================================================================

@app.get(
    "/analyze/{component_id}"
)
async def analyze_component_get(
    component_id: str
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
# ANALYZE COMPONENT - POST
# =============================================================================

@app.post("/analyze")
async def analyze_component_post(
    request: AnalyzeRequest
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
# REFRESH
# =============================================================================

@app.post("/refresh")
async def refresh():

    global ANALYSIS_CACHE

    ANALYSIS_CACHE = {}

    return {

        "status": "success",

        "message":
            "Analysis cache cleared.",
    }


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )
