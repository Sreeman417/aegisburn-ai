"""
AEGISBURN AI — Parameter-Aware 168h Drift Prediction

Module B:
Predict Value_168h from early burn-in measurements.

One model is trained for each:
    component_type + parameter_name

Input features:
    value_0h
    value_24h

Target:
    value_168h

The registry uses a canonical model key so filenames and
runtime lookups remain consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# FEATURES
# ============================================================

PREDICTION_FEATURES: List[str] = [
    "value_0h",
    "value_24h",
]

TARGET_COLUMN = "value_168h"


# ============================================================
# RESULT
# ============================================================

@dataclass
class PredictionResult:
    predicted_168h: float
    component_type: str
    parameter_name: str
    model_name: str


# ============================================================
# SINGLE PREDICTOR
# ============================================================

class ParameterDriftPredictor:

    def __init__(
        self,
        random_state: int = 42,
    ) -> None:

        self.random_state = random_state

        self.model: Optional[Pipeline] = None

        self.model_name: Optional[str] = None

        self.component_type: Optional[str] = None

        self.parameter_name: Optional[str] = None

        self.feature_names = (
            PREDICTION_FEATURES.copy()
        )

        self.training_count: int = 0

        self.mae: Optional[float] = None

        self.rmse: Optional[float] = None

        self.r2: Optional[float] = None

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        df: pd.DataFrame,
        component_type: Optional[str] = None,
        parameter_name: Optional[str] = None,
    ) -> "ParameterDriftPredictor":

        required = [
            *self.feature_names,
            TARGET_COLUMN,
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing prediction columns: {missing}"
            )

        if component_type is not None:
            self.component_type = (
                str(component_type).strip()
            )

        if parameter_name is not None:
            self.parameter_name = (
                str(parameter_name).strip()
            )

        train_df = df.copy()

        X = train_df[
            self.feature_names
        ].copy()

        y = train_df[
            TARGET_COLUMN
        ].copy()

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        y = pd.to_numeric(
            y,
            errors="coerce",
        )

        valid = (
            X.notna().all(axis=1)
            & y.notna()
        )

        X = X.loc[valid]
        y = y.loc[valid]

        if len(X) < 20:
            raise ValueError(
                "Not enough valid rows for prediction model. "
                f"Found {len(X)}, need at least 20."
            )

        # ----------------------------------------------------
        # Candidate models
        # ----------------------------------------------------

        models = {
            "LinearRegression": Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "model",
                        LinearRegression(),
                    ),
                ]
            ),

            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=300,
                random_state=self.random_state,
                n_jobs=-1,
                min_samples_leaf=2,
            ),

            "GradientBoostingRegressor": GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state,
            ),
        }

        best_model = None
        best_model_name = None
        best_mae = float("inf")

        best_rmse = None
        best_r2 = None

        # ----------------------------------------------------
        # Evaluate candidates
        # ----------------------------------------------------
        #
        # For this prototype, use an internal chronological-style
        # holdout rather than evaluating on exactly the same rows
        # used for fitting.
        #
        # This is still synthetic-data evaluation and should later
        # be replaced with a proper lot-aware train/test split.
        # ----------------------------------------------------

        split_index = int(
            len(X) * 0.80
        )

        if split_index < 10:
            split_index = len(X) - 10

        X_train = X.iloc[
            :split_index
        ]

        y_train = y.iloc[
            :split_index
        ]

        X_test = X.iloc[
            split_index:
        ]

        y_test = y.iloc[
            split_index:
        ]

        for name, model in models.items():

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                predictions,
            )

            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions,
                    )
                )
            )

            r2 = r2_score(
                y_test,
                predictions,
            )

            if mae < best_mae:

                best_mae = mae

                best_rmse = rmse

                best_r2 = r2

                best_model = model

                best_model_name = name

        if best_model is None:
            raise RuntimeError(
                "Unable to select a prediction model."
            )

        # ----------------------------------------------------
        # Refit selected model on all available data
        # ----------------------------------------------------

        best_model.fit(
            X,
            y,
        )

        self.model = best_model

        self.model_name = best_model_name

        self.training_count = len(X)

        self.mae = float(
            best_mae
        )

        self.rmse = float(
            best_rmse
        )

        self.r2 = float(
            best_r2
        )

        return self

    # ========================================================
    # VALIDATION
    # ========================================================

    def _check_fitted(self) -> None:

        if self.model is None:
            raise RuntimeError(
                "Prediction model is not fitted."
            )

    def _prepare_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        missing = [
            column
            for column in self.feature_names
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing prediction features: {missing}"
            )

        X = df[
            self.feature_names
        ].copy()

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        for column in self.feature_names:

            if X[column].isna().any():

                median_value = X[
                    column
                ].median()

                if pd.isna(
                    median_value
                ):

                    median_value = 0.0

                X[column] = X[
                    column
                ].fillna(
                    median_value
                )

        return X

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        self._check_fitted()

        X = self._prepare_features(
            df
        )

        predictions = self.model.predict(
            X
        )

        return np.asarray(
            predictions,
            dtype=float,
        )

    # ========================================================
    # ONE COMPONENT
    # ========================================================

    def predict_component(
        self,
        row: pd.Series | Dict,
    ) -> PredictionResult:

        if isinstance(
            row,
            pd.Series,
        ):

            record = row.to_dict()

        else:

            record = dict(row)

        component_type = str(
            record.get(
                "component_type",
                "",
            )
        ).strip()

        parameter_name = str(
            record.get(
                "parameter_name",
                "",
            )
        ).strip()

        if not component_type:

            raise ValueError(
                "component_type is required."
            )

        if not parameter_name:

            raise ValueError(
                "parameter_name is required."
            )

        prediction = self.predict(
            pd.DataFrame(
                [record]
            )
        )[0]

        return PredictionResult(
            predicted_168h=float(
                prediction
            ),
            component_type=component_type,
            parameter_name=parameter_name,
            model_name=(
                self.model_name
                or "unknown"
            ),
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path: str | Path,
    ) -> None:

        self._check_fitted()

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "model": self.model,

            "model_name": self.model_name,

            "component_type":
                self.component_type,

            "parameter_name":
                self.parameter_name,

            "feature_names":
                self.feature_names,

            "training_count":
                self.training_count,

            "mae":
                self.mae,

            "rmse":
                self.rmse,

            "r2":
                self.r2,

            "random_state":
                self.random_state,
        }

        joblib.dump(
            payload,
            path,
        )

    # ========================================================
    # LOAD
    # ========================================================

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "ParameterDriftPredictor":

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(
                f"Prediction model not found: {path}"
            )

        payload = joblib.load(
            path
        )

        predictor = cls(
            random_state=payload.get(
                "random_state",
                42,
            )
        )

        predictor.model = payload[
            "model"
        ]

        predictor.model_name = payload.get(
            "model_name"
        )

        predictor.component_type = payload.get(
            "component_type"
        )

        predictor.parameter_name = payload.get(
            "parameter_name"
        )

        predictor.feature_names = payload.get(
            "feature_names",
            PREDICTION_FEATURES.copy(),
        )

        predictor.training_count = payload.get(
            "training_count",
            0,
        )

        predictor.mae = payload.get(
            "mae"
        )

        predictor.rmse = payload.get(
            "rmse"
        )

        predictor.r2 = payload.get(
            "r2"
        )

        return predictor


# ============================================================
# MODEL REGISTRY
# ============================================================

class PredictionModelRegistry:

    def __init__(self) -> None:

        self.models: Dict[
            str,
            ParameterDriftPredictor
        ] = {}

    # ========================================================
    # CANONICAL KEY
    # ========================================================

    @staticmethod
    def make_key(
        component_type: str,
        parameter_name: str,
    ) -> str:

        component_type = str(
            component_type
        ).strip()

        parameter_name = str(
            parameter_name
        ).strip()

        return (
            f"{component_type}"
            f"__"
            f"{parameter_name}"
        )

    # ========================================================
    # FILE KEY
    # ========================================================

    @staticmethod
    def make_filename_key(
        component_type: str,
        parameter_name: str,
    ) -> str:

        component_type = str(
            component_type
        ).strip()

        parameter_name = str(
            parameter_name
        ).strip()

        # Filenames use underscores instead of spaces.
        component_type = (
            component_type
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        parameter_name = (
            parameter_name
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        return (
            f"{component_type}"
            f"__"
            f"{parameter_name}"
        )

    # ========================================================
    # ADD
    # ========================================================

    def add(
        self,
        component_type: str,
        parameter_name: str,
        predictor: ParameterDriftPredictor,
    ) -> None:

        key = self.make_key(
            component_type,
            parameter_name,
        )

        self.models[key] = predictor

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        component_type: str,
        parameter_name: str,
    ) -> ParameterDriftPredictor:

        canonical_key = self.make_key(
            component_type,
            parameter_name,
        )

        # Exact canonical lookup.
        if canonical_key in self.models:

            return self.models[
                canonical_key
            ]

        # Some previously saved models used underscores
        # in the registry key. Support that format too.
        filename_key = (
            self.make_filename_key(
                component_type,
                parameter_name,
            )
        )

        if filename_key in self.models:

            return self.models[
                filename_key
            ]

        # Compare normalized forms defensively.
        normalized_requested = (
            canonical_key
            .lower()
            .replace(" ", "_")
            .strip()
        )

        for key, model in self.models.items():

            normalized_key = (
                key
                .lower()
                .replace(" ", "_")
                .strip()
            )

            if normalized_key == normalized_requested:

                return model

        available = ", ".join(
            sorted(
                self.models.keys()
            )
        )

        raise KeyError(
            f"No prediction model available for "
            f"{canonical_key}. "
            f"Available models: {available}"
        )

    # ========================================================
    # FIT ALL
    # ========================================================

    def fit_all(
        self,
        df: pd.DataFrame,
        random_state: int = 42,
    ) -> "PredictionModelRegistry":

        required = [
            "component_type",
            "parameter_name",
            *PREDICTION_FEATURES,
            TARGET_COLUMN,
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing prediction columns: {missing}"
            )

        self.models = {}

        grouped = df.groupby(
            [
                "component_type",
                "parameter_name",
            ],
            dropna=False,
        )

        for (
            component_type,
            parameter_name,
        ), group in grouped:

            component_type = str(
                component_type
            ).strip()

            parameter_name = str(
                parameter_name
            ).strip()

            if not component_type:
                continue

            if not parameter_name:
                continue

            predictor = ParameterDriftPredictor(
                random_state=random_state
            )

            predictor.fit(
                group,
                component_type=component_type,
                parameter_name=parameter_name,
            )

            self.add(
                component_type,
                parameter_name,
                predictor,
            )

        if not self.models:

            raise ValueError(
                "No prediction models were trained."
            )

        return self

    # ========================================================
    # PREDICT ONE COMPONENT
    # ========================================================

    def predict_component(
        self,
        row: pd.Series | Dict,
    ) -> PredictionResult:

        if isinstance(
            row,
            pd.Series,
        ):

            record = row.to_dict()

        else:

            record = dict(row)

        component_type = str(
            record.get(
                "component_type",
                "",
            )
        ).strip()

        parameter_name = str(
            record.get(
                "parameter_name",
                "",
            )
        ).strip()

        predictor = self.get(
            component_type,
            parameter_name,
        )

        return predictor.predict_component(
            record
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        directory: str | Path,
    ) -> None:

        directory = Path(
            directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_rows = []

        for key, predictor in self.models.items():

            component_type = (
                predictor.component_type
                or ""
            )

            parameter_name = (
                predictor.parameter_name
                or ""
            )

            filename_key = (
                self.make_filename_key(
                    component_type,
                    parameter_name,
                )
            )

            model_path = (
                directory
                / f"{filename_key}.joblib"
            )

            predictor.save(
                model_path
            )

            metadata_rows.append(
                {
                    "model_key": key,

                    "filename":
                        model_path.name,

                    "component_type":
                        component_type,

                    "parameter_name":
                        parameter_name,

                    "model":
                        predictor.model_name,

                    "mae":
                        predictor.mae,

                    "rmse":
                        predictor.rmse,

                    "r2":
                        predictor.r2,
                }
            )

        metadata = pd.DataFrame(
            metadata_rows
        )

        metadata.to_csv(
            directory / "registry.csv",
            index=False,
        )

    # ========================================================
    # LOAD
    # ========================================================

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "PredictionModelRegistry":

        directory = Path(
            directory
        )

        if not directory.exists():

            raise FileNotFoundError(
                f"Prediction model directory "
                f"not found: {directory}"
            )

        registry = cls()

        model_files = sorted(
            directory.glob(
                "*.joblib"
            )
        )

        if not model_files:

            raise FileNotFoundError(
                f"No prediction .joblib models "
                f"found in {directory}"
            )

        for model_path in model_files:

            predictor = (
                ParameterDriftPredictor.load(
                    model_path
                )
            )

            component_type = (
                predictor.component_type
            )

            parameter_name = (
                predictor.parameter_name
            )

            if not component_type:
                raise ValueError(
                    f"Missing component_type "
                    f"in {model_path}"
                )

            if not parameter_name:
                raise ValueError(
                    f"Missing parameter_name "
                    f"in {model_path}"
                )

            registry.add(
                component_type,
                parameter_name,
                predictor,
            )

        return registry


# ============================================================
# TRAINING CONVENIENCE FUNCTION
# ============================================================

def train_prediction_models(
    df: pd.DataFrame,
    output_dir: str | Path = "models/prediction",
    random_state: int = 42,
) -> PredictionModelRegistry:

    registry = (
        PredictionModelRegistry()
    )

    registry.fit_all(
        df,
        random_state=random_state,
    )

    registry.save(
        output_dir
    )

    return registry


# ============================================================
# LOADING CONVENIENCE FUNCTION
# ============================================================

def load_prediction_models(
    model_dir: str | Path = "models/prediction",
) -> PredictionModelRegistry:

    return (
        PredictionModelRegistry.load(
            model_dir
        )
    )


# ============================================================
# STANDALONE
# ============================================================

if __name__ == "__main__":

    print(
        "AegisBurn AI prediction module loaded."
    )

    print(
        "Prediction features:",
        PREDICTION_FEATURES,
    )