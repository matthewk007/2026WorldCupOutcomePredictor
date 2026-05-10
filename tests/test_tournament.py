from pathlib import Path

import pandas as pd

from src.tournament import load_fixture_board, build_knockout_bracket, build_group_standings


def test_load_fixture_board_requires_file(tmp_path):
    missing = tmp_path / "official_2026_fixtures.csv"

    try:
        load_fixture_board(missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        assert True


def test_load_fixture_board_reads_required_columns(tmp_path):
    fixture_file = tmp_path / "official_2026_fixtures.csv"
    pd.DataFrame(
        {
            "stage": ["Group Stage"],
            "group": ["A"],
            "match_number": [1],
            "home_team": ["Canada"],
            "away_team": ["Mexico"],
        }
    ).to_csv(fixture_file, index=False)

    board = load_fixture_board(fixture_file)

    assert list(board.columns) == ["stage", "group", "match_number", "home_team", "away_team"]
    assert board.loc[0, "home_team"] == "Canada"


def test_build_group_standings_orders_teams_by_points_and_goal_difference():
    results = pd.DataFrame(
        {
            "group": ["A", "A", "A"],
            "home_team": ["Canada", "Mexico", "Canada"],
            "away_team": ["Mexico", "Canada", "USA"],
            "home_score": [2, 1, 0],
            "away_score": [1, 1, 2],
        }
    )

    standings = build_group_standings(results)

    assert list(standings["team"])[:2] == ["Canada", "USA"]
    assert standings.loc[standings["team"] == "Canada", "points"].iloc[0] == 4


def test_build_knockout_bracket_uses_group_winners_and_runners_up():
    standings = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "team": ["Canada", "Mexico", "Brazil", "Argentina"],
            "position": [1, 2, 1, 2],
        }
    )

    bracket = build_knockout_bracket(standings)

    assert "Round of 32" in bracket["stage"].unique()
    assert len(bracket) == 2


def test_build_knockout_bracket_returns_full_round_of_32():
    standings = pd.DataFrame(
        {
            "group": [
                "A", "A", "A",
                "B", "B", "B",
                "C", "C", "C",
                "D", "D", "D",
                "E", "E", "E",
                "F", "F", "F",
                "G", "G", "G",
                "H", "H", "H",
                "I", "I", "I",
                "J", "J", "J",
                "K", "K", "K",
                "L", "L", "L",
            ],
            "team": [
                "A1", "A2", "A3",
                "B1", "B2", "B3",
                "C1", "C2", "C3",
                "D1", "D2", "D3",
                "E1", "E2", "E3",
                "F1", "F2", "F3",
                "G1", "G2", "G3",
                "H1", "H2", "H3",
                "I1", "I2", "I3",
                "J1", "J2", "J3",
                "K1", "K2", "K3",
                "L1", "L2", "L3",
            ],
            "position": [1, 2, 3] * 12,
        }
    )

    bracket = build_knockout_bracket(standings)

    assert len(bracket) == 16
