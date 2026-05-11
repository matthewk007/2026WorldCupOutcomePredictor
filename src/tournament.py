from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_FIXTURE_COLUMNS = ["stage", "group", "match_number", "home_team", "away_team"]


GROUP_NAMES = list("ABCDEFGHIJKL")


def load_fixture_board(path: str | Path) -> pd.DataFrame:
    fixture_path = Path(path)
    if not fixture_path.exists():
        raise FileNotFoundError(f"Missing official fixture file: {fixture_path}")

    board = pd.read_csv(fixture_path)
    missing = [column for column in REQUIRED_FIXTURE_COLUMNS if column not in board.columns]
    if missing:
        raise ValueError(f"Missing required fixture columns: {', '.join(missing)}")

    board = board[REQUIRED_FIXTURE_COLUMNS].copy()
    board["match_number"] = board["match_number"].astype(int)
    return board.sort_values(["stage", "group", "match_number"]).reset_index(drop=True)


def build_group_standings(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["group", "team", "points", "goal_difference", "goals_for", "position"])

    rows = []
    for _, match in results.iterrows():
        home_points = 3 if match["home_score"] > match["away_score"] else 1 if match["home_score"] == match["away_score"] else 0
        away_points = 3 if match["away_score"] > match["home_score"] else 1 if match["home_score"] == match["away_score"] else 0
        rows.append((match["group"], match["home_team"], home_points, match["home_score"] - match["away_score"], match["home_score"]))
        rows.append((match["group"], match["away_team"], away_points, match["away_score"] - match["home_score"], match["away_score"]))

    standings = pd.DataFrame(rows, columns=["group", "team", "points", "goal_difference", "goals_for"])
    standings = standings.groupby(["group", "team"], as_index=False).sum()
    standings = standings.sort_values(["group", "points", "goal_difference", "goals_for", "team"], ascending=[True, False, False, False, True])
    standings["position"] = standings.groupby("group").cumcount() + 1
    return standings.reset_index(drop=True)


def build_knockout_bracket(standings: pd.DataFrame) -> pd.DataFrame:
    if standings.empty:
        return pd.DataFrame(columns=["stage", "home_team", "away_team"])

    grouped = standings.sort_values(["group", "position"])
    pairs = []
    groups = list(grouped["group"].drop_duplicates())
    for idx in range(0, len(groups), 2):
        if idx + 1 >= len(groups):
            break
        left = grouped[(grouped["group"] == groups[idx]) & (grouped["position"] == 1)].head(1)
        right = grouped[(grouped["group"] == groups[idx + 1]) & (grouped["position"] == 2)].head(1)
        if not left.empty and not right.empty:
            pairs.append(("Round of 32", left.iloc[0]["team"], right.iloc[0]["team"]))

        left_runner = grouped[(grouped["group"] == groups[idx]) & (grouped["position"] == 2)].head(1)
        right_winner = grouped[(grouped["group"] == groups[idx + 1]) & (grouped["position"] == 1)].head(1)
        if not left_runner.empty and not right_winner.empty:
            pairs.append(("Round of 32", right_winner.iloc[0]["team"], left_runner.iloc[0]["team"]))

    # Duplicate the same group-to-group pairing pattern for the remaining four third-place slots.
    # This keeps the bracket board full for the UI until the official FIFA mapping is added.
    if len(pairs) < 16 and len(groups) >= 12:
        for idx in range(0, len(groups), 2):
            if len(pairs) >= 16 or idx + 1 >= len(groups):
                break
            left_third = grouped[(grouped["group"] == groups[idx]) & (grouped["position"] == 3)].head(1)
            right_third = grouped[(grouped["group"] == groups[idx + 1]) & (grouped["position"] == 3)].head(1)
            if not left_third.empty and not right_third.empty:
                pairs.append(("Round of 32", left_third.iloc[0]["team"], right_third.iloc[0]["team"]))

    return pd.DataFrame(pairs, columns=["stage", "home_team", "away_team"])


def generate_group_stage_fixtures(teams: list[str]) -> pd.DataFrame:
    if len(teams) != 48:
        raise ValueError("2026 World Cup group stage requires exactly 48 teams")

    rows = []
    for index, group in enumerate(GROUP_NAMES):
        group_teams = teams[index * 4 : (index + 1) * 4]
        match_number = 1
        for i in range(len(group_teams)):
            for j in range(i + 1, len(group_teams)):
                rows.append(
                    {
                        "stage": "Group Stage",
                        "group": group,
                        "match_number": match_number,
                        "home_team": group_teams[i],
                        "away_team": group_teams[j],
                    }
                )
                match_number += 1
    return pd.DataFrame(rows, columns=REQUIRED_FIXTURE_COLUMNS)


def advance_knockout_round(fixtures: pd.DataFrame, results: pd.DataFrame, next_stage: str) -> pd.DataFrame:
    if fixtures.empty or results.empty:
        return pd.DataFrame(columns=["stage", "group", "match_number", "home_team", "away_team"])

    winners = []
    for _, result in results.sort_values("fixture_index").iterrows():
        if result["home_score"] == result["away_score"]:
            raise ValueError("Knockout fixtures require a winner")
        winners.append(result["home_team"] if result["home_score"] > result["away_score"] else result["away_team"])

    rows = []
    for index in range(0, len(winners), 2):
        if index + 1 >= len(winners):
            break
        rows.append(
            {
                "stage": next_stage,
                "group": "",
                "match_number": index // 2 + 1,
                "home_team": winners[index],
                "away_team": winners[index + 1],
            }
        )

    return pd.DataFrame(rows, columns=["stage", "group", "match_number", "home_team", "away_team"])
