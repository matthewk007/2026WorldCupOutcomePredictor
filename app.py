from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.bundle import load_kaggle_bundle
from src.config import DEFAULT_MODEL_DIR
from src.model import ensure_artifacts_from_bundle
from src.predict import get_model_dir, predict_match


def main():
    st.title("2026 World Cup Predictor")
    st.write("Predict win/draw/loss and an expected score from Kaggle-trained models.")

    home_team = st.text_input("Home team", value="Brazil")
    away_team = st.text_input("Away team", value="Argentina")
    match_date = st.date_input("Match date")
    tournament = st.text_input("Tournament", value="World Cup")
    model_dir = st.text_input("Model directory", value=str(get_model_dir(DEFAULT_MODEL_DIR)))

    bundle = load_kaggle_bundle(Path("data_sources.toml"))
    ensure_artifacts_from_bundle(model_dir, Path("data_sources.toml"))

    if st.button("Predict"):
        if not Path(model_dir).exists():
            st.error("Model artifacts not found. Train the models first.")
            return

        prediction = predict_match(
            home_team,
            away_team,
            model_dir=model_dir,
            match_date=match_date,
            tournament=tournament,
            rankings=bundle.rankings,
        )
        st.subheader("Prediction")
        st.write(f"Outcome: {prediction['predicted_outcome']}")
        st.write(f"Expected score: {prediction['expected_score']}")
        st.json(prediction["outcome_probs"])


if __name__ == "__main__":
    main()
