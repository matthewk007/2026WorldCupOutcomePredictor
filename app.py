from __future__ import annotations

from pathlib import Path

import streamlit as st
import pandas as pd

from src.bundle import load_kaggle_bundle
from src.config import DEFAULT_MODEL_DIR
from src.model import ensure_artifacts_from_bundle
from src.predict import get_model_dir, predict_match

st.set_page_config(
    page_title="2026 World Cup Predictor",
    page_icon="🏆",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Inject Shadcn-inspired CSS (Option 3 style polish)
st.markdown(
    """
    <style>
    /* Clean typography & card styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }
    p {
        color: #64748b; /* muted-foreground */
    }
    .stButton>button {
        width: 100%;
        font-weight: 600;
        border-radius: 0.5rem;
        height: 2.75rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
    st.title("🏆 World Cup Predictor")
    st.write("Predict match outcomes and scores using historical Kaggle data and ML models.")

    # Initialize data & models on startup
    with st.spinner("Initializing models..."):
        bundle = load_kaggle_bundle(Path("data_sources.toml"))
        ensure_artifacts_from_bundle(str(get_model_dir(DEFAULT_MODEL_DIR)), Path("data_sources.toml"))

    st.markdown("### Match Details")
    
    # Use columns for a cleaner form layout
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.text_input("Home Team", value="Brazil", placeholder="e.g. Brazil")
    with col2:
        away_team = st.text_input("Away Team", value="Argentina", placeholder="e.g. Argentina")
        
    col3, col4 = st.columns(2)
    with col3:
        match_date = st.date_input("Match Date")
    with col4:
        tournament = st.selectbox(
            "Tournament", 
            ["World Cup", "Friendly", "Copa America", "UEFA Euro", "Gold Cup", "AFCON"]
        )

    # Advanced Settings in expander to keep UI clean
    with st.expander("⚙️ Advanced Settings"):
        model_dir = st.text_input("Model Directory", value=str(get_model_dir(DEFAULT_MODEL_DIR)))

    st.markdown("---")

    if st.button("Predict Outcome", type="primary"):
        if not Path(model_dir).exists():
            st.error("Model artifacts not found. Train the models first.")
            return

        with st.spinner("Analyzing match..."):
            try:
                prediction = predict_match(
                    home_team,
                    away_team,
                    model_dir=model_dir,
                    match_date=match_date,
                    tournament=tournament,
                    rankings=bundle.rankings,
                )
                
                # Results Section - clean metrics layout
                st.markdown("### Match Prediction")
                
                # Main predictions
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    st.metric("Predicted Outcome", prediction['predicted_outcome'].upper())
                with r_col2:
                    st.metric("Expected Score", prediction['expected_score'])
                
                # Probability distribution
                st.markdown("#### Outcome Probabilities")
                probs = prediction["outcome_probs"]
                
                p_col1, p_col2, p_col3 = st.columns(3)
                with p_col1:
                    st.metric("Home Win", f"{probs.get('home_win', 0)*100:.1f}%")
                with p_col2:
                    st.metric("Draw", f"{probs.get('draw', 0)*100:.1f}%")
                with p_col3:
                    st.metric("Away Win", f"{probs.get('away_win', 0)*100:.1f}%")
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
