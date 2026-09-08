"""
Parameter-specific anomaly detection for burn-in screening.

Module A:
Detect abnormal early-life component behavior using only:
    - value_0h
    - value_24h
    - early drift features
    - lot/reference-normalized features

Important:
    value_96h and value_168h are NOT used by the anomaly model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Features allowed for EARLY anomaly screening
# ---------------------------------------------------------------------

EARLY_FEATURES: List[str] = [
    "value_0h",
    "value_24h",
    "drift_0_24",
    "slope_early",
    "ratio_24_0",
    "z_score_0h",
    "z_score_24h",
    "z_score_slope_early",
]


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------

@dataclass
class AnomalyResult:
    """
    Result returned for one component.
    """

    is_anomaly: bool
    anomaly_score: float
    raw_score: float
    parameter_name: str
    component_type: str


# ---------------------------------------------------------------------
# Single parameter/type anomaly detector
# ---------------------------------------------------------------------

class ParameterAnomalyDetector:
    """
    Isolation Forest anomaly detector for one component-type /
    parameter family.

    Example:
        Logic IC + Iddq
        Memory IC + Standby Current
        ADC + Leakage Current
        Driver IC + Propagation Delay
    """

    def __init__(
        self,
        contamination: float = 0.10,
        n_estimators: int = 300,
        random_state: int = 42,
    ) -> None:
        if not 0 < contamination < 0.5:
            raise ValueError("contamination must be between 0 and 0.5")

        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.model: Optional[Pipeline] = None
        self.is_fitted: bool = False

        self.component_type: Optional[str] = None
        self.parameter_name: Optional[str] = None

        self.feature_names: List[str] = EARLY_FEATURES.copy()

        self.training_count: int = 0
        self.training_anomaly_rate: Optional[float] = None

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        component_type: Optional[str] = None,
        parameter_name: Optional[str] = None,
    ) -> "ParameterAnomalyDetector":
        """
        Train Isolation Forest on early-life features only.
        """

        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing anomaly features: {missing}"
            )

        train_df = df.copy()

        # Store family identifiers if supplied.
        if component_type is not None:
            self.component_type = str(component_type).strip()

        if parameter_name is not None:
            self.parameter_name = str(parameter_name).strip()

        # Keep only valid numeric rows.
        X = train_df[self.feature_names].copy()

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()

        if len(X) < 20:
            raise ValueError(
                "Not enough valid rows to train anomaly detector. "
                f"Found {len(X)}, need at least 20."
            )

        # Isolation Forest becomes more stable with explicit scaling.
        self.model = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "isolation_forest",
                    IsolationForest(
                        n_estimators=self.n_estimators,
                        contamination=self.contamination,
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        self.model.fit(X)

        predictions = self.model.predict(X)

        self.training_count = len(X)
        self.training_anomaly_rate = float(
            np.mean(predictions == -1)
        )

        self.is_fitted = True

        return self

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted or self.model is None:
            raise RuntimeError(
                "Anomaly detector is not fitted. "
                "Call fit() or load a trained model first."
            )

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_names if c not in df.columns]

        if missing:
            raise ValueError(
                f"Missing anomaly features: {missing}"
            )

        X = df[self.feature_names].copy()

        X = X.replace([np.inf, -np.inf], np.nan)

        # Fill missing numeric values using feature medians.
        for column in self.feature_names:
            if X[column].isna().any():
                median_value = X[column].median()

                if pd.isna(median_value):
                    median_value = 0.0

                X[column] = X[column].fillna(median_value)

        return X

    # -----------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns:
            1  = normal
           -1  = anomaly
        """

        self._check_fitted()

        X = self._prepare_features(df)

        return self.model.predict(X)

    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns Isolation Forest decision-function scores.

        Higher = more normal.
        Lower  = more anomalous.
        """

        self._check_fitted()

        X = self._prepare_features(df)

        return self.model.decision_function(X)

    def anomaly_index(self, df: pd.DataFrame) -> np.ndarray:
        """
        Converts Isolation Forest scores into a 0-100 anomaly index.

        Higher = more anomalous.

        This is NOT a probability.
        """

        raw_scores = self.predict_score(df)

        # IsolationForest decision_function is generally centered around
        # zero. Convert it to a bounded anomaly-like index.
        #
        # Negative scores -> increasing anomaly index
        # Positive scores -> decreasing anomaly index
        index = 50.0 - (raw_scores * 100.0)

        index = np.clip(index, 0.0, 100.0)

        return index

    # -----------------------------------------------------------------
    # Component-level result
    # -----------------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add anomaly prediction fields to a dataframe.
        """

        result = df.copy()

        predictions = self.predict(result)
        raw_scores = self.predict_score(result)
        indices = self.anomaly_index(result)

        result["anomaly_flag"] = (predictions == -1).astype(int)
        result["anomaly_label"] = np.where(
            predictions == -1,
            "ANOMALY",
            "NORMAL",
        )
        result["anomaly_raw_score"] = raw_scores
        result["anomaly_index"] = indices

        return result

    # -----------------------------------------------------------------
    # Save / load
    # -----------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        self._check_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "component_type": self.component_type,
            "parameter_name": self.parameter_name,
            "feature_names": self.feature_names,
            "training_count": self.training_count,
            "training_anomaly_rate": self.training_anomaly_rate,
        }

        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "ParameterAnomalyDetector":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Anomaly model not found: {path}"
            )

        payload = joblib.load(path)

        detector = cls(
            contamination=payload.get(
                "contamination",
                0.10,
            ),
            n_estimators=payload.get(
                "n_estimators",
                300,
            ),
            random_state=payload.get(
                "random_state",
                42,
            ),
        )

        detector.model = payload["model"]

        detector.component_type = payload.get(
            "component_type"
        )

        detector.parameter_name = payload.get(
            "parameter_name"
        )

        detector.feature_names = payload.get(
            "feature_names",
            EARLY_FEATURES.copy(),
        )

        detector.training_count = payload.get(
            "training_count",
            0,
        )

        detector.training_anomaly_rate = payload.get(
            "training_anomaly_rate"
        )

        detector.is_fitted = True

        return detector


# ---------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------

class ParameterModelRegistry:
    """
    Stores one anomaly model per component-type / parameter family.

    Example keys:
        Logic IC__Iddq
        Memory IC__Standby Current
        ADC__Leakage Current
        Driver IC__Propagation Delay
    """

    def __init__(self) -> None:
        self.models: Dict[str, ParameterAnomalyDetector] = {}

    # -----------------------------------------------------------------
    # Key handling
    # -----------------------------------------------------------------

    @staticmethod
    def make_key(
        component_type: str,
        parameter_name: str,
    ) -> str:
        """
        Create one canonical model key.

        Whitespace is normalized so training and inference produce
        exactly the same key.
        """

        component_type = str(component_type).strip()
        parameter_name = str(parameter_name).strip()

        return f"{component_type}__{parameter_name}"

    # -----------------------------------------------------------------
    # Add / get
    # -----------------------------------------------------------------

    def add(
        self,
        component_type: str,
        parameter_name: str,
        detector: ParameterAnomalyDetector,
    ) -> None:
        key = self.make_key(
            component_type,
            parameter_name,
        )

        self.models[key] = detector

    def get(
        self,
        component_type: str,
        parameter_name: str,
    ) -> ParameterAnomalyDetector:
        key = self.make_key(
            component_type,
            parameter_name,
        )

        # Exact lookup first.
        if key in self.models:
            return self.models[key]

        # Extra defensive lookup in case an old model dictionary contains
        # accidental whitespace.
        normalized_key = self.make_key(
            component_type.strip(),
            parameter_name.strip(),
        )

        if normalized_key in self.models:
            return self.models[normalized_key]

        available = ", ".join(sorted(self.models.keys()))

        raise KeyError(
            f"No anomaly model available for {key}. "
            f"Available models: {available}"
        )

    # -----------------------------------------------------------------
    # Fit all families
    # -----------------------------------------------------------------

    def fit_all(
        self,
        df: pd.DataFrame,
        contamination: float = 0.10,
        n_estimators: int = 300,
        random_state: int = 42,
    ) -> "ParameterModelRegistry":
        """
        Train one anomaly model per component_type + parameter_name.
        """

        required = [
            "component_type",
            "parameter_name",
            *EARLY_FEATURES,
        ]

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns for anomaly training: {missing}"
            )

        self.models = {}

        grouped = df.groupby(
            ["component_type", "parameter_name"],
            dropna=False,
        )

        for (
            component_type,
            parameter_name,
        ), group in grouped:

            component_type = str(component_type).strip()
            parameter_name = str(parameter_name).strip()

            if not component_type or not parameter_name:
                continue

            detector = ParameterAnomalyDetector(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=random_state,
            )

            detector.fit(
                group,
                component_type=component_type,
                parameter_name=parameter_name,
            )

            self.add(
                component_type,
                parameter_name,
                detector,
            )

        if not self.models:
            raise ValueError(
                "No anomaly models were trained."
            )

        return self

    # -----------------------------------------------------------------
    # Predict one component
    # -----------------------------------------------------------------

    def predict_component(
        self,
        row: pd.Series | Dict,
    ) -> AnomalyResult:
        """
        Analyze one component record.
        """

        if isinstance(row, pd.Series):
            record = row.to_dict()
        else:
            record = dict(row)

        component_type = str(
            record.get("component_type", "")
        ).strip()

        parameter_name = str(
            record.get("parameter_name", "")
        ).strip()

        if not component_type:
            raise ValueError(
                "component_type is required."
            )

        if not parameter_name:
            raise ValueError(
                "parameter_name is required."
            )

        detector = self.get(
            component_type,
            parameter_name,
        )

        single_df = pd.DataFrame([record])

        prediction = detector.predict(single_df)[0]
        raw_score = detector.predict_score(single_df)[0]
        anomaly_index = detector.anomaly_index(single_df)[0]

        return AnomalyResult(
            is_anomaly=bool(prediction == -1),
            anomaly_score=float(anomaly_index),
            raw_score=float(raw_score),
            parameter_name=parameter_name,
            component_type=component_type,
        )

    # -----------------------------------------------------------------
    # Analyze dataframe
    # -----------------------------------------------------------------

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the correct parameter-specific anomaly detector to every
        row in the dataframe.
        """

        required = [
            "component_type",
            "parameter_name",
            *EARLY_FEATURES,
        ]

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns for anomaly analysis: {missing}"
            )

        result = df.copy()

        result["anomaly_flag"] = 0
        result["anomaly_label"] = "NORMAL"
        result["anomaly_raw_score"] = np.nan
        result["anomaly_index"] = np.nan

        # Process each family with its matching model.
        grouped = result.groupby(
            ["component_type", "parameter_name"],
            dropna=False,
        )

        for (
            component_type,
            parameter_name,
        ), indices in grouped.groups.items():

            component_type = str(component_type).strip()
            parameter_name = str(parameter_name).strip()

            detector = self.get(
                component_type,
                parameter_name,
            )

            family_df = result.loc[
                indices
            ].copy()

            family_result = detector.analyze(
                family_df
            )

            result.loc[
                indices,
                "anomaly_flag"
            ] = family_result[
                "anomaly_flag"
            ].values

            result.loc[
                indices,
                "anomaly_label"
            ] = family_result[
                "anomaly_label"
            ].values

            result.loc[
                indices,
                "anomaly_raw_score"
            ] = family_result[
                "anomaly_raw_score"
            ].values

            result.loc[
                indices,
                "anomaly_index"
            ] = family_result[
                "anomaly_index"
            ].values

        result["anomaly_flag"] = (
            result["anomaly_flag"]
            .fillna(0)
            .astype(int)
        )

        result["anomaly_index"] = (
            result["anomaly_index"]
            .fillna(0.0)
            .astype(float)
        )

        return result

    # Alias useful for backend code.
    analyze = analyze_dataframe

    # -----------------------------------------------------------------
    # Registry info
    # -----------------------------------------------------------------

    def available_models(self) -> List[str]:
        return sorted(self.models.keys())

    def summary(self) -> pd.DataFrame:
        """
        Return model metadata.
        """

        rows = []

        for key, detector in self.models.items():
            rows.append(
                {
                    "model_key": key,
                    "component_type": detector.component_type,
                    "parameter_name": detector.parameter_name,
                    "contamination": detector.contamination,
                    "n_estimators": detector.n_estimators,
                    "training_count": detector.training_count,
                    "training_anomaly_rate": (
                        detector.training_anomaly_rate
                    ),
                }
            )

        return pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Save registry
    # -----------------------------------------------------------------

    def save(
        self,
        directory: str | Path,
    ) -> None:
        """
        Save every parameter-specific model as a separate .joblib file.

        IMPORTANT:
        Spaces in parameter names are preserved.
        Only path separators are replaced because they would break
        filenames.
        """

        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save a registry summary too.
        metadata_rows = []

        for key, detector in self.models.items():

            safe_name = key.replace(
                "/",
                "_",
            ).replace(
                "\\",
                "_",
            )

            model_path = directory / (
                f"{safe_name}.joblib"
            )

            detector.save(model_path)

            metadata_rows.append(
                {
                    "model_key": key,
                    "filename": model_path.name,
                    "component_type": (
                        detector.component_type
                    ),
                    "parameter_name": (
                        detector.parameter_name
                    ),
                }
            )

        metadata = pd.DataFrame(
            metadata_rows
        )

        metadata.to_csv(
            directory / "registry.csv",
            index=False,
        )

    # -----------------------------------------------------------------
    # Load registry
    # -----------------------------------------------------------------

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "ParameterModelRegistry":
        """
        Load all .joblib parameter-specific anomaly models.
        """

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Anomaly model directory not found: {directory}"
            )

        registry = cls()

        model_files = sorted(
            directory.glob("*.joblib")
        )

        if not model_files:
            raise FileNotFoundError(
                f"No anomaly .joblib models found in {directory}"
            )

        for model_path in model_files:

            detector = ParameterAnomalyDetector.load(
                model_path
            )

            component_type = (
                detector.component_type
            )
            parameter_name = (
                detector.parameter_name
            )

            if not component_type or not parameter_name:
                raise ValueError(
                    f"Model metadata missing in {model_path}"
                )

            registry.add(
                component_type,
                parameter_name,
                detector,
            )

        return registry


# ---------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------

def train_anomaly_models(
    df: pd.DataFrame,
    output_dir: str | Path = "models/anomaly",
    contamination: float = 0.10,
    n_estimators: int = 300,
    random_state: int = 42,
) -> ParameterModelRegistry:
    """
    Train and save all parameter-specific anomaly models.
    """

    registry = ParameterModelRegistry()

    registry.fit_all(
        df=df,
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    registry.save(output_dir)

    return registry


def load_anomaly_models(
    model_dir: str | Path = "models/anomaly",
) -> ParameterModelRegistry:
    """
    Load saved parameter-specific anomaly models.
    """

    return ParameterModelRegistry.load(model_dir)


# ---------------------------------------------------------------------
# Simple standalone test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Parameter anomaly detection module loaded successfully.")
    print(f"Early features: {EARLY_FEATURES}")
