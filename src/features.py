from __future__ import annotations

import numpy as np
import pandas as pd


def _result_from_goal_diff(goal_diff: int) -> str:
    if goal_diff > 0:
        return "home_win"
    if goal_diff < 0:
        return "away_win"
    return "draw"


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)
    out["goal_diff"] = out["home_score"] - out["away_score"]
    out["result"] = out["goal_diff"].map(_result_from_goal_diff)
    out["match_year"] = out["date"].dt.year
    out["match_month"] = out["date"].dt.month
    out["match_dayofweek"] = out["date"].dt.dayofweek
    out["total_goals"] = out["home_score"] + out["away_score"]
    out["home_form_points"] = _rolling_form(out, "home_team", "home_score", "away_score")
    out["away_form_points"] = _rolling_form(out, "away_team", "away_score", "home_score")
    if "home_rank" not in out.columns:
        out["home_rank"] = np.nan
    if "away_rank" not in out.columns:
        out["away_rank"] = np.nan
    return out


def attach_rankings(matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    matches_out = matches.copy()
    rankings_out = rankings.copy()

    home_rankings = rankings_out.rename(columns={"team": "home_team", "rank": "home_rank"})[["date", "home_team", "home_rank"]]
    away_rankings = rankings_out.rename(columns={"team": "away_team", "rank": "away_rank"})[["date", "away_team", "away_rank"]]

    matches_out = pd.merge_asof(
        matches_out.sort_values("date"),
        home_rankings.sort_values("date"),
        on="date",
        by="home_team",
        direction="backward",
    )
    matches_out = pd.merge_asof(
        matches_out.sort_values("date"),
        away_rankings.sort_values("date"),
        on="date",
        by="away_team",
        direction="backward",
    )
    return matches_out


def make_inference_row(
    home_team: str,
    away_team: str,
    match_date=None,
    tournament: str = "World Cup",
    ranking_snapshot: dict[str, int] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "home_team": home_team,
                "away_team": away_team,
                "tournament": tournament,
                "date": pd.to_datetime(match_date) if match_date else pd.Timestamp.today(),
            }
        ]
    )
    frame["match_year"] = frame["date"].dt.year
    frame["match_month"] = frame["date"].dt.month
    frame["match_dayofweek"] = frame["date"].dt.dayofweek
    frame["home_form_points"] = 0
    frame["away_form_points"] = 0
    ranking_snapshot = ranking_snapshot or {}
    frame["home_rank"] = ranking_snapshot.get(home_team)
    frame["away_rank"] = ranking_snapshot.get(away_team)
    return frame.drop(columns=["date"])


def _rolling_form(df: pd.DataFrame, team_col: str, team_score_col: str, opp_score_col: str) -> pd.Series:
    points = []
    history = {}

    for _, row in df.iterrows():
        team = row[team_col]
        points.append(sum(history.get(team, [])[-3:]))

        team_points = 3 if row[team_score_col] > row[opp_score_col] else 1 if row[team_score_col] == row[opp_score_col] else 0
        history.setdefault(team, []).append(team_points)

    return pd.Series(points, index=df.index)
