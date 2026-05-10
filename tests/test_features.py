import pandas as pd

from src.features import build_training_frame


def test_build_training_frame_adds_target_columns():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-06-14"]),
            "home_team": ["Russia"],
            "away_team": ["Saudi Arabia"],
            "home_score": [5],
            "away_score": [0],
        }
    )

    out = build_training_frame(df)
    assert "result" in out.columns
    assert "goal_diff" in out.columns
    assert out.loc[0, "result"] == "home_win"
    assert out.loc[0, "goal_diff"] == 5
