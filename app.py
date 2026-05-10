from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.bundle import load_kaggle_bundle
from src.config import DEFAULT_FIXTURE_PATH, DEFAULT_MODEL_DIR
from src.model import ensure_artifacts_from_bundle
from src.predict import get_model_dir, predict_match
from src.tournament import build_group_standings, build_knockout_bracket, load_fixture_board


st.set_page_config(
    page_title="2026 World Cup Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-weight: 800; letter-spacing: -0.025em; }
    .fixture-card { border: 1px solid #e2e8f0; border-radius: 14px; padding: 1rem; margin-bottom: 0.75rem; background: white; }
    .fixture-meta { color: #64748b; font-size: 0.9rem; }
    .stage-title { margin-top: 1.5rem; margin-bottom: 0.5rem; }
    .stButton > button { width: 100%; border-radius: 0.65rem; height: 2.75rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _get_team_list(bundle) -> list[str]:
    teams = set()
    if bundle.rankings is not None and "team" in bundle.rankings.columns:
        teams.update(bundle.rankings["team"].dropna().unique())
    if bundle.matches is not None:
        teams.update(bundle.matches.get("home_team", pd.Series(dtype=str)).dropna().unique())
        teams.update(bundle.matches.get("away_team", pd.Series(dtype=str)).dropna().unique())
    return sorted(teams)


def _load_results_from_state(fixtures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, fixture in fixtures.iterrows():
        key = f"fixture_{idx}"
        score = st.session_state.get(key)
        if not score:
            continue
        rows.append(
            {
                "group": fixture["group"],
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "home_score": int(score[0]),
                "away_score": int(score[1]),
            }
        )
    return pd.DataFrame(rows)


def _render_fixture_row(idx: int, fixture: pd.Series, model_dir: Path, rankings) -> None:
    st.markdown(
        f"""
        <div class="fixture-card">
            <div><strong>{fixture['home_team']}</strong> vs <strong>{fixture['away_team']}</strong></div>
            <div class="fixture-meta">{fixture['stage']} · Group {fixture['group']} · Match {int(fixture['match_number'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    score_key = f"fixture_{idx}"
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        home_score = st.number_input(
            f"{fixture['home_team']} goals",
            min_value=0,
            max_value=20,
            value=st.session_state.get(score_key, (0, 0))[0],
            key=f"{score_key}_home",
        )
    with col2:
        away_score = st.number_input(
            f"{fixture['away_team']} goals",
            min_value=0,
            max_value=20,
            value=st.session_state.get(score_key, (0, 0))[1],
            key=f"{score_key}_away",
        )
    with col3:
        if st.button(f"Save {fixture['home_team']} vs {fixture['away_team']}", key=f"save_{idx}"):
            st.session_state[score_key] = (int(home_score), int(away_score))
            st.success("Prediction saved")

    if st.session_state.get(score_key):
        prediction = predict_match(
            fixture["home_team"],
            fixture["away_team"],
            model_dir=model_dir,
            match_date=None,
            tournament="World Cup",
            rankings=rankings,
        )
        st.caption(f"Model outlook: {prediction['predicted_outcome']} | Expected score: {prediction['expected_score']}")


def main():
    st.title("🏆 2026 World Cup Predictor")
    st.write("Predict every match in tournament order, then advance into the knockout bracket.")

    if not DEFAULT_FIXTURE_PATH.exists():
        st.error(f"Missing official fixture file: {DEFAULT_FIXTURE_PATH}")
        raise FileNotFoundError(f"Missing official fixture file: {DEFAULT_FIXTURE_PATH}")

    bundle = load_kaggle_bundle(Path("data_sources.toml"))
    ensure_artifacts_from_bundle(str(get_model_dir(DEFAULT_MODEL_DIR)), Path("data_sources.toml"))

    fixture_path = DEFAULT_FIXTURE_PATH
    if not fixture_path.exists():
        st.error(f"Missing official fixture file: {fixture_path}")
        st.stop()

    fixtures = load_fixture_board(fixture_path)
    model_dir = get_model_dir(DEFAULT_MODEL_DIR)

    group_fixtures = fixtures[fixtures["stage"] == "Group Stage"].copy()
    knockout_fixtures = fixtures[fixtures["stage"] != "Group Stage"].copy()

    group_tab, knockout_tab = st.tabs(["Group Stage", "Knockout Stage"])

    with group_tab:
        st.markdown("### Group Stage Fixtures")
        for group in sorted(group_fixtures["group"].dropna().unique()):
            st.markdown(f"<div class='stage-title'><strong>Group {group}</strong></div>", unsafe_allow_html=True)
            group_rows = group_fixtures[group_fixtures["group"] == group]
            for idx, fixture in group_rows.iterrows():
                _render_fixture_row(idx, fixture, model_dir, bundle.rankings)

        results = _load_results_from_state(group_fixtures)
        if not results.empty:
            standings = build_group_standings(results)
            st.markdown("### Current Group Standings")
            st.dataframe(standings, use_container_width=True)

    with knockout_tab:
        st.markdown("### Knockout Stage")
        results = _load_results_from_state(group_fixtures)
        if results.empty:
            st.info("Predict the group stage first to unlock the knockout bracket.")
            st.stop()

        standings = build_group_standings(results)
        bracket = build_knockout_bracket(standings)
        if bracket.empty:
            st.info("The knockout bracket will appear after the group stage standings are complete.")
            st.stop()

        st.dataframe(bracket, use_container_width=True)


if __name__ == "__main__":
    main()
