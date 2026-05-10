from __future__ import annotations

import datetime
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

def get_unique_teams(bundle) -> list[str]:
    teams = set()
    if bundle.rankings is not None and "team" in bundle.rankings.columns:
        teams.update(bundle.rankings["team"].dropna().unique())
    if bundle.matches is not None:
        if "home_team" in bundle.matches.columns:
            teams.update(bundle.matches["home_team"].dropna().unique())
        if "away_team" in bundle.matches.columns:
            teams.update(bundle.matches["away_team"].dropna().unique())
            
    # List of expected/qualified teams for 2026 World Cup (48 teams)
    # Covering common naming conventions from the dataset
    projected_2026_teams = {
        "Argentina", "Australia", "Austria", "Belgium", "Brazil", "Cameroon", 
        "Canada", "Colombia", "Costa Rica", "Croatia", "Denmark", "Ecuador", 
        "Egypt", "England", "France", "Germany", "Iran", "IR Iran", "Iraq", "Italy", 
        "Ivory Coast", "Côte d'Ivoire", "Jamaica", "Japan", "Mali", "Mexico", "Morocco", 
        "Netherlands", "New Zealand", "Nigeria", "Panama", "Peru", "Poland", 
        "Portugal", "Qatar", "Saudi Arabia", "Senegal", "Serbia", "South Korea", "Korea Republic",
        "Spain", "Sweden", "Switzerland", "Tunisia", "Ukraine", "United States", "USA", 
        "Uruguay", "Uzbekistan", "Wales", "Algeria", "Hungary", "Turkey", "Türkiye", "Norway"
    }
    
    # Filter only teams that are expected in 2026 AND exist in the dataset
    valid_2026_teams = sorted(list(teams.intersection(projected_2026_teams)))
    
    return valid_2026_teams

def main():
    st.title("🏆 2026 World Cup Predictor")
    st.write("Predict match outcomes for the upcoming 2026 World Cup.")

    # Initialize data & models on startup
    with st.spinner("Initializing models..."):
        bundle = load_kaggle_bundle(Path("data_sources.toml"))
        ensure_artifacts_from_bundle(str(get_model_dir(DEFAULT_MODEL_DIR)), Path("data_sources.toml"))
        team_list = get_unique_teams(bundle)
        
        # Fallback if empty for some reason
        if not team_list:
            team_list = ["Brazil", "Argentina", "France", "England", "Spain", "Germany", "USA", "Mexico"]

    st.markdown("### Match Details")
    
    # Pre-select some default teams if available
    default_home = "USA" if "USA" in team_list else team_list[0]
    default_away = "Mexico" if "Mexico" in team_list else (team_list[1] if len(team_list) > 1 else team_list[0])

    # Use columns for a cleaner form layout
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", options=team_list, index=team_list.index(default_home))
    with col2:
        away_team = st.selectbox("Away Team", options=team_list, index=team_list.index(default_away))

    # We enforce 2026 World Cup logic automatically without asking the user
    match_date = datetime.date(2026, 6, 11)  # Kickoff date for 2026 World Cup
    tournament = "World Cup"

    # Advanced Settings in expander to keep UI clean
    with st.expander("⚙️ Advanced Settings"):
        model_dir = st.text_input(
            "Model Directory", 
            value=str(get_model_dir(DEFAULT_MODEL_DIR)),
            disabled=True,
            help="The directory where trained ML model artifacts are stored. This is managed automatically."
        )

    st.markdown("---")

    if st.button("Predict Outcome", type="primary"):
        if home_team == away_team:
            st.error("Please select two different teams.")
            return
            
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
                    st.metric(f"{home_team} Win", f"{probs.get('home_win', 0)*100:.1f}%")
                with p_col2:
                    st.metric("Draw", f"{probs.get('draw', 0)*100:.1f}%")
                with p_col3:
                    st.metric(f"{away_team} Win", f"{probs.get('away_win', 0)*100:.1f}%")
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
