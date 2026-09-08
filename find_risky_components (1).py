from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
ANOMALY_MODEL_DIR = MODELS_DIR / "anomaly"
PREDICTION_MODEL_DIR = MODELS_DIR / "prediction"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# =============================================================================
# IMPORT PROJECT MODULES
# =============================================================================

try:
    from src.anomaly_detection import AnomalyModelRegistry
except Exception as error:
    print(f"WARNING: Could not import AnomalyModelRegistry: {error}")
    AnomalyModelRegistry = None


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
# GLOBAL MODELS
# =============================================================================

ANOMALY_MODELS: Any = None
PREDICTION_MODELS: Any = None
RISK_ENGINE: Any = None


# =============================================================================
# COLUMN NORMALIZATION
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


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def is_valid_dataset(path: Path) -> bool:

    try:

        sample = pd.read_csv(
            path,
            nrows=5
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

        parts_lower = [
            part.lower()
            for part in path.parts
        ]

        if "models" in parts_lower:
            continue

        filename_lower = path.name.lower()

        if "registry" in filename_lower:
            continue

        if is_valid_dataset(path):

            valid_files.append(path)

    if not valid_files:

        raise FileNotFoundError(
            "NO VALID AEGISBURN DATASET FOUND."
        )

    valid_files.sort(
        key=lambda path: path.stat().st_size,
        reverse=True
    )

    selected_path = valid_files[0]

    print(
        f"DATASET SELECTED: {selected_path}"
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
        f"Rows loaded: {len(df):,}"
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

        compact = column.replace("_", "")

        if column in aliases:

            rename_map[column] = aliases[column]

        elif compact in aliases:

            rename_map[column] = aliases[compact]

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
            f"Missing columns: {missing_columns}"
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

        if column in df.columns:

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

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
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

    df = df.reset_index(
        drop=True
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
            np.nan
        )

        .fillna(0.0)
    )

    return features


# =============================================================================
# LOAD ANOMALY MODELS
# =============================================================================

def load_anomaly_models():

    if AnomalyModelRegistry is None:

        return None

    if not ANOMALY_MODEL_DIR.exists():

        print(
            "WARNING: Anomaly model directory not found."
        )

        return None

    try:

        models = AnomalyModelRegistry.load(
            str(ANOMALY_MODEL_DIR)
        )

        print(
            "Anomaly models loaded successfully."
        )

        return models

    except Exception as error:

        print(
            f"Could not load anomaly models: {error}"
        )

        return None


# =============================================================================
# LOAD PREDICTION MODELS
# =============================================================================

def load_prediction_models():

    if PredictionModelRegistry is None:

        return None

    if not PREDICTION_MODEL_DIR.exists():

        print(
            "WARNING: Prediction model directory not found."
        )

        return None

    try:

        models = PredictionModelRegistry.load(
            str(PREDICTION_MODEL_DIR)
        )

        print(
            "Prediction models loaded successfully."
        )

        return models

    except Exception as error:

        print(
            f"Could not load prediction models: {error}"
        )

        return None


# =============================================================================
# INITIALIZE RISK ENGINE
# =============================================================================

def initialize_risk_engine():

    if RiskEngine is None:

        return None

    try:

        engine = RiskEngine()

        print(
            "Risk engine initialized successfully."
        )

        return engine

    except Exception as error:

        print(
            f"Could not initialize RiskEngine: {error}"
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

        "anomaly_score": 0.0,

        "anomaly_index": 0.0,
    }

    if ANOMALY_MODELS is None:

        return default

    try:

        result = (
            ANOMALY_MODELS
            .predict_component(row)
        )

        if isinstance(result, dict):

            flag = result.get(
                "anomaly_flag",
                result.get(
                    "flag",
                    0
                )
            )

            score = result.get(
                "anomaly_score",
                result.get(
                    "score",
                    0.0
                )
            )

            index = result.get(
                "anomaly_index",
                score
            )

        else:

            flag = getattr(
                result,
                "anomaly_flag",
                getattr(
                    result,
                    "flag",
                    0
                )
            )

            score = getattr(
                result,
                "anomaly_score",
                getattr(
                    result,
                    "score",
                    0.0
                )
            )

            index = getattr(
                result,
                "anomaly_index",
                score
            )

        return {

            "anomaly_flag": int(
                bool(flag)
            ),

            "anomaly_score": float(
                score
            ),

            "anomaly_index": float(
                index
            ),
        }

    except Exception as error:

        print(
            f"Anomaly error for "
            f"{row.get('component_id')}: "
            f"{error}"
        )

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
            0.0
        )
    )

    fallback = pd.to_numeric(
        fallback,
        errors="coerce"
    )

    if pd.isna(fallback):

        fallback = row.get(
            "value_24h",
            0.0
        )

    fallback = float(
        fallback
    )

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
            "predicted_168h"
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

    except Exception:

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
                0.0
            )
        )
    )

    return max(
        slope,
        0.001
    )


# =============================================================================
# RISK CALCULATION
# =============================================================================

def calculate_risk(
    row: pd.Series,
    anomaly_result: dict,
    predicted_168h: float,
    safety_slope: float
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
                0
            )
        )

        risk_row["anomaly_score"] = float(
            anomaly_result.get(
                "anomaly_score",
                0.0
            )
        )

        risk_row["anomaly_index"] = float(
            anomaly_result.get(
                "anomaly_index",
                0.0
            )
        )

        result = (
            RISK_ENGINE
            .evaluate_component(
                risk_row,
                float(predicted_168h),
                float(safety_slope)
            )
        )

        if isinstance(result, dict):

            return {

                "risk_score": float(
                    result.get(
                        "risk_score",
                        0.0
                    )
                ),

                "risk_level": str(
                    result.get(
                        "risk_level",
                        "NORMAL"
                    )
                ),

                "early_drift_index": float(
                    result.get(
                        "early_drift_index",
                        0.0
                    )
                ),

                "future_drift_index": float(
                    result.get(
                        "future_drift_index",
                        0.0
                    )
                ),

                "limit_utilization": float(
                    result.get(
                        "limit_utilization",
                        0.0
                    )
                ),

                "engineering_limit": result.get(
                    "engineering_limit",
                    None
                ),

                "reasons": result.get(
                    "reasons",
                    []
                ),
            }

        return {

            "risk_score": float(
                getattr(
                    result,
                    "risk_score",
                    0.0
                )
            ),

            "risk_level": str(
                getattr(
                    result,
                    "risk_level",
                    "NORMAL"
                )
            ),

            "early_drift_index": float(
                getattr(
                    result,
                    "early_drift_index",
                    0.0
                )
            ),

            "future_drift_index": float(
                getattr(
                    result,
                    "future_drift_index",
                    0.0
                )
            ),

            "limit_utilization": float(
                getattr(
                    result,
                    "limit_utilization",
                    0.0
                )
            ),

            "engineering_limit": getattr(
                result,
                "engineering_limit",
                None
            ),

            "reasons": list(
                getattr(
                    result,
                    "reasons",
                    []
                )
            ),
        }

    except Exception as error:

        print(
            f"Risk calculation error: {error}"
        )

        return default


# =============================================================================
# CHECK IF COMPONENT IS RISKY
# =============================================================================

def is_risky(
    anomaly_result: dict,
    risk_result: dict
) -> bool:

    anomaly_flag = int(
        anomaly_result.get(
            "anomaly_flag",
            0
        )
    )

    risk_level = str(
        risk_result.get(
            "risk_level",
            "NORMAL"
        )
    ).upper()

    risk_score = float(
        risk_result.get(
            "risk_score",
            0.0
        )
    )

    # Component is considered risky if:
    #
    # 1. Anomaly model detects anomaly
    # OR
    # 2. Risk level is not NORMAL
    # OR
    # 3. Risk score is greater than zero

    return (

        anomaly_flag == 1

        or risk_level != "NORMAL"

        or risk_score > 0.0
    )


# =============================================================================
# ANALYZE ALL COMPONENTS
# =============================================================================

def analyze_all_components(
    feature_data: pd.DataFrame
):

    component_ids = (
        feature_data[
            "component_id"
        ]
        .astype(str)
        .unique()
    )

    total_components = len(
        component_ids
    )

    risky_components = []

    print()
    print("=" * 70)
    print("ANALYZING ALL COMPONENTS FOR RISK")
    print("=" * 70)

    print(
        f"Total component IDs to analyze: "
        f"{total_components}"
    )

    print()
    print(
        "Risky components will be printed LIVE "
        "as soon as they are detected..."
    )

    print("=" * 70)

    for number, component_id in enumerate(
        component_ids,
        start=1
    ):

        try:

            matches = feature_data[
                feature_data[
                    "component_id"
                ].astype(str)
                == component_id
            ]

            if matches.empty:

                continue

            row = (
                matches
                .iloc[0]
                .copy()
            )

            # ---------------------------------------------------------
            # ANOMALY
            # ---------------------------------------------------------

            anomaly_result = calculate_anomaly(
                row
            )

            row["anomaly_flag"] = int(
                anomaly_result[
                    "anomaly_flag"
                ]
            )

            row["anomaly_score"] = float(
                anomaly_result[
                    "anomaly_score"
                ]
            )

            row["anomaly_index"] = float(
                anomaly_result[
                    "anomaly_index"
                ]
            )

            # ---------------------------------------------------------
            # PREDICTION
            # ---------------------------------------------------------

            predicted_168h = (
                calculate_prediction(
                    row
                )
            )

            # ---------------------------------------------------------
            # SAFETY SLOPE
            # ---------------------------------------------------------

            safety_slope = (
                calculate_safety_slope(
                    row
                )
            )

            # ---------------------------------------------------------
            # RISK
            # ---------------------------------------------------------

            risk_result = calculate_risk(
                row,
                anomaly_result,
                predicted_168h,
                safety_slope
            )

            # ---------------------------------------------------------
            # LIVE RISK DETECTION
            # ---------------------------------------------------------

            if is_risky(
                anomaly_result,
                risk_result
            ):

                result = {

                    "component_id":
                        component_id,

                    "component_type":
                        row.get(
                            "component_type",
                            ""
                        ),

                    "parameter_name":
                        row.get(
                            "parameter_name",
                            ""
                        ),

                    "anomaly_flag":
                        anomaly_result[
                            "anomaly_flag"
                        ],

                    "anomaly_score":
                        anomaly_result[
                            "anomaly_score"
                        ],

                    "anomaly_index":
                        anomaly_result[
                            "anomaly_index"
                        ],

                    "predicted_168h":
                        predicted_168h,

                    "risk_score":
                        risk_result[
                            "risk_score"
                        ],

                    "risk_level":
                        risk_result[
                            "risk_level"
                        ],

                    "early_drift_index":
                        risk_result[
                            "early_drift_index"
                        ],

                    "future_drift_index":
                        risk_result[
                            "future_drift_index"
                        ],

                    "limit_utilization":
                        risk_result[
                            "limit_utilization"
                        ],

                    "reasons":
                        risk_result[
                            "reasons"
                        ],
                }

                risky_components.append(
                    result
                )

                print()
                print(
                    "🚨 RISK / DEFECT DETECTED!"
                )

                print(
                    f"Component ID: "
                    f"{component_id}"
                )

                print(
                    f"Type: "
                    f"{result['component_type']}"
                )

                print(
                    f"Parameter: "
                    f"{result['parameter_name']}"
                )

                print(
                    f"Anomaly Flag: "
                    f"{result['anomaly_flag']}"
                )

                print(
                    f"Anomaly Score: "
                    f"{result['anomaly_score']:.6f}"
                )

                print(
                    f"Risk Score: "
                    f"{result['risk_score']:.6f}"
                )

                print(
                    f"Risk Level: "
                    f"{result['risk_level']}"
                )

                print(
                    f"Predicted 168h: "
                    f"{result['predicted_168h']:.6f}"
                )

                print(
                    f"Reasons: "
                    f"{result['reasons']}"
                )

                print(
                    "-" * 70
                )

            # ---------------------------------------------------------
            # PROGRESS
            # ---------------------------------------------------------

            if number % 100 == 0:

                print(
                    f"[Progress] "
                    f"{number}/{total_components} "
                    f"checked | "
                    f"{len(risky_components)} risky found"
                )

        except Exception as error:

            print(
                f"ERROR analyzing "
                f"{component_id}: {error}"
            )

            continue

    return risky_components


# =============================================================================
# MAIN
# =============================================================================

def main():

    global ANOMALY_MODELS
    global PREDICTION_MODELS
    global RISK_ENGINE

    print()
    print("=" * 70)
    print("AEGISBURN AI - RISKY COMPONENT FINDER")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------------------------------

    print("Loading dataset...")

    component_data = load_dataset()

    print(
        "Creating features..."
    )

    feature_data = create_features(
        component_data
    )

    # -------------------------------------------------------------------------
    # LOAD MODELS
    # -------------------------------------------------------------------------

    print(
        "Loading anomaly models..."
    )

    ANOMALY_MODELS = (
        load_anomaly_models()
    )

    print(
        "Loading prediction models..."
    )

    PREDICTION_MODELS = (
        load_prediction_models()
    )

    print(
        "Initializing risk engine..."
    )

    RISK_ENGINE = (
        initialize_risk_engine()
    )

    # -------------------------------------------------------------------------
    # ANALYZE ALL COMPONENTS
    # -------------------------------------------------------------------------

    risky_components = (
        analyze_all_components(
            feature_data
        )
    )

    # -------------------------------------------------------------------------
    # FINAL RESULTS
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Total risky components found: "
        f"{len(risky_components)}"
    )

    # -------------------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------------------

    if risky_components:

        output_file = (
            BASE_DIR
            / "risky_components.csv"
        )

        results_df = pd.DataFrame(
            risky_components
        )

        results_df.to_csv(
            output_file,
            index=False
        )

        print()

        print(
            "Results saved to:"
        )

        print(
            output_file
        )

    else:

        print()

        print(
            "No risky components were detected."
        )

    print("=" * 70)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Analysis stopped by user."
        )

    except Exception as error:

        print()
        print(
            "FATAL ERROR:"
        )

        print(error)

        traceback.print_exc()
