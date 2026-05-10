from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.bundle import ranking_snapshot_before
from src.features import make_inference_row
from src.model import ensure_artifacts, load_artifacts


def format_prediction(outcome_probs, expected_score):
    predicted_outcome = max(outcome_probs, key=outcome_probs.get)
    return {
        "predicted_outcome": predicted_outcome,
        "outcome_probs": outcome_probs,
        "expected_score": f"{expected_score[0]}-{expected_score[1]}",
    }


def predict_match(
    home_team: str,
    away_team: str,
    model_dir: str | Path,
    match_date=None,
    tournament: str = "World Cup",
    rankings=None,
):
    artifacts = ensure_artifacts(model_dir)
    classifier, regressor = load_artifacts(model_dir)
    ranking_snapshot = {}
    if rankings is not None and match_date is not None:
        ranking_snapshot = ranking_snapshot_before(rankings, match_date)

    row = make_inference_row(
        home_team,
        away_team,
        match_date=match_date,
        tournament=tournament,
        ranking_snapshot=ranking_snapshot,
    )
    probs = classifier.predict_proba(row)
    labels = list(classifier.classes_)
    outcome_probs = {label: float(prob) for label, prob in zip(labels, probs[0])}

    goal_diff = float(regressor.predict(row)[0])
    home_goals = max(0, round((goal_diff + 2) / 2))
    away_goals = max(0, round((2 - goal_diff) / 2))
    return format_prediction(outcome_probs, (home_goals, away_goals))


def get_model_dir(default: str | Path) -> Path:
    return Path(os.getenv("MODEL_DIR", default))
