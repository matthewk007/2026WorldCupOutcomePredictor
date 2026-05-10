from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.constants import OUTCOME_LABELS
from src.features import build_training_frame


FEATURE_COLUMNS = ["home_team", "away_team", "match_year", "match_month", "match_dayofweek", "home_form_points", "away_form_points", "home_rank", "away_rank"]
TARGET_COLUMN = "result"
REG_TARGET_COLUMN = "goal_diff"


@dataclass
class ModelArtifacts:
    classifier_path: Path
    regressor_path: Path


def fallback_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-06-14",
                    "2018-06-15",
                    "2018-06-16",
                    "2018-06-17",
                    "2018-06-18",
                    "2018-06-19",
                ]
            ),
            "home_team": ["Russia", "Egypt", "Brazil", "Germany", "France", "Spain"],
            "away_team": ["Saudi Arabia", "Uruguay", "Switzerland", "Mexico", "Australia", "Iran"],
            "home_score": [5, 1, 1, 0, 2, 3],
            "away_score": [0, 0, 1, 1, 1, 0],
            "home_rank": [0, 0, 0, 0, 0, 0],
            "away_rank": [0, 0, 0, 0, 0, 0],
        }
    )


def _preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    categorical = ["home_team", "away_team"]
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0))]),
                numeric_features,
            ),
        ]
    )


def _available_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            continue
        if column in {"home_rank", "away_rank"} and not df[column].notna().any():
            continue
        columns.append(column)
    return columns


def train_models(df: pd.DataFrame, output_dir) -> ModelArtifacts:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frame = build_training_frame(df)
    feature_columns = _available_feature_columns(frame)
    X = frame[feature_columns]
    y_class = frame[TARGET_COLUMN]
    y_reg = frame[REG_TARGET_COLUMN]

    numeric_features = [
        column
        for column in ["match_year", "match_month", "match_dayofweek", "home_form_points", "away_form_points", "home_rank", "away_rank"]
        if column in feature_columns
    ]

    classifier = Pipeline(
        steps=[
            ("preprocess", _preprocessor(numeric_features)),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    )
    regressor = Pipeline(
        steps=[
            ("preprocess", _preprocessor(numeric_features)),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )

    classifier.fit(X, y_class)
    regressor.fit(X, y_reg)

    classifier_path = out / "classifier.joblib"
    regressor_path = out / "regressor.joblib"
    dump({"model": classifier, "labels": OUTCOME_LABELS}, classifier_path)
    dump({"model": regressor}, regressor_path)

    return ModelArtifacts(classifier_path=classifier_path, regressor_path=regressor_path)


def ensure_artifacts(model_dir) -> ModelArtifacts:
    model_dir = Path(model_dir)
    classifier_path = model_dir / "classifier.joblib"
    regressor_path = model_dir / "regressor.joblib"

    if classifier_path.exists() and regressor_path.exists():
        return ModelArtifacts(classifier_path=classifier_path, regressor_path=regressor_path)

    return train_models(fallback_training_frame(), model_dir)


def load_artifacts(model_dir) -> tuple[Pipeline, Pipeline]:
    model_dir = Path(model_dir)
    classifier_bundle = load(model_dir / "classifier.joblib")
    regressor_bundle = load(model_dir / "regressor.joblib")
    return classifier_bundle["model"], regressor_bundle["model"]
