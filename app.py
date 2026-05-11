from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.bundle import load_kaggle_bundle
from src.config import DEFAULT_FIXTURE_PATH, DEFAULT_MODEL_DIR
from src.model import ensure_artifacts_from_bundle
from src.predict import get_model_dir, predict_match
from src.tournament import (
    advance_knockout_round,
    build_group_standings,
    build_knockout_bracket,
    load_fixture_board,
)


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


def _ensure_session_keys(fixtures: pd.DataFrame) -> None:
    for idx in fixtures.index:
        st.session_state.setdefault(f"fixture_{idx}", None)


def _save_score(idx: int, home_score: int, away_score: int) -> None:
    st.session_state[f"fixture_{idx}"] = (home_score, away_score)


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

    current_score = st.session_state.get(f"fixture_{idx}") or (0, 0)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        home_score = st.number_input(
            f"{fixture['home_team']} goals",
            min_value=0,
            max_value=20,
            value=int(current_score[0]),
            key=f"{idx}_home",
        )
    with col2:
        away_score = st.number_input(
            f"{fixture['away_team']} goals",
            min_value=0,
            max_value=20,
            value=int(current_score[1]),
            key=f"{idx}_away",
        )
    with col3:
        if st.button(f"Save {fixture['home_team']} vs {fixture['away_team']}", key=f"save_{idx}"):
            _save_score(idx, int(home_score), int(away_score))
            st.success("Prediction saved")

    if st.session_state.get(f"fixture_{idx}") is not None:
        prediction = predict_match(
            fixture["home_team"],
            fixture["away_team"],
            model_dir=model_dir,
            match_date=None,
            tournament="World Cup",
            rankings=rankings,
        )
        st.caption(f"Model outlook: {prediction['predicted_outcome']} | Expected score: {prediction['expected_score']}")


def _results_frame(fixtures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, fixture in fixtures.iterrows():
        score = st.session_state.get(f"fixture_{idx}")
        if score is None:
            continue
        rows.append(
            {
                "fixture_index": idx,
                "group": fixture["group"],
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "home_score": int(score[0]),
                "away_score": int(score[1]),
            }
        )
    return pd.DataFrame(rows)


def main():
    st.title("🏆 2026 World Cup Predictor")
    st.write("Predict the full tournament in stage order: group stage first, then knockout rounds.")

    bundle = load_kaggle_bundle(Path("data_sources.toml"))
    ensure_artifacts_from_bundle(str(get_model_dir(DEFAULT_MODEL_DIR)), Path("data_sources.toml"))
    model_dir = get_model_dir(DEFAULT_MODEL_DIR)

    teams = _get_team_list(bundle)
    if len(teams) < 48:
        st.error("Need at least 48 teams from the dataset to generate the 2026 group board.")
        st.stop()

    if not DEFAULT_FIXTURE_PATH.exists():
        st.error(f"Missing official fixture file: {DEFAULT_FIXTURE_PATH}")
        raise FileNotFoundError(f"Missing official fixture file: {DEFAULT_FIXTURE_PATH}")

    fixtures = load_fixture_board(DEFAULT_FIXTURE_PATH)

    _ensure_session_keys(fixtures)

    group_fixtures = fixtures[fixtures["stage"] == "Group Stage"].copy()
    group_results = _results_frame(group_fixtures)

    standings = build_group_standings(group_results) if not group_results.empty else pd.DataFrame()
    knockout_board = build_knockout_bracket(standings) if not standings.empty else pd.DataFrame()

    group_tab, knockout_tab = st.tabs(["Group Stage", "Knockout Stage"])

    with group_tab:
        st.markdown("### Group Stage Fixtures")
        for group in sorted(group_fixtures["group"].dropna().unique()):
            st.markdown(f"<div class='stage-title'><strong>Group {group}</strong></div>", unsafe_allow_html=True)
            group_rows = group_fixtures[group_fixtures["group"] == group]
            for idx, fixture in group_rows.iterrows():
                _render_fixture_row(idx, fixture, model_dir, bundle.rankings)

        if not standings.empty:
            st.markdown("### Current Group Standings")
            st.dataframe(standings, use_container_width=True)

    with knockout_tab:
        st.markdown("### Knockout Stage")
        if standings.empty:
            st.info("Complete some group stage predictions to unlock the knockout bracket.")
            st.stop()

        st.dataframe(knockout_board, use_container_width=True)

        if not knockout_board.empty:
            st.markdown("### Advance Winners")
            st.caption("Enter knockout results in order to generate the next round.")

            round_results = []
            for idx, fixture in knockout_board.iterrows():
                st.markdown(
                    f"""
                    <div class="fixture-card">
                        <div><strong>{fixture['home_team']}</strong> vs <strong>{fixture['away_team']}</strong></div>
                        <div class="fixture-meta">{fixture['stage']} · Match {idx + 1}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col1, col2 = st.columns(2)
                with col1:
                    hs = st.number_input(f"{fixture['home_team']} knockout goals", min_value=0, max_value=20, value=0, key=f"ko_{idx}_home")
                with col2:
                    as_ = st.number_input(f"{fixture['away_team']} knockout goals", min_value=0, max_value=20, value=0, key=f"ko_{idx}_away")
                round_results.append({
                    "fixture_index": idx,
                    "home_team": fixture["home_team"],
                    "away_team": fixture["away_team"],
                    "home_score": int(hs),
                    "away_score": int(as_),
                })

            if st.button("Generate Next Round"):
                next_round = advance_knockout_round(
                    knockout_board,
                    pd.DataFrame(round_results),
                    next_stage="Quarterfinal",
                )
                st.dataframe(next_round, use_container_width=True)


if __name__ == "__main__":
    main()
