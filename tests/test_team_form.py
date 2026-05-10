import pandas as pd

from src.features import build_training_frame


def test_build_training_frame_adds_team_form_features():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-06-14", "2018-06-18", "2018-06-22"]),
            "home_team": ["Russia", "Russia", "Russia"],
            "away_team": ["Saudi Arabia", "Egypt", "Uruguay"],
            "home_score": [5, 3, 0],
            "away_score": [0, 1, 1],
        }
    )

    out = build_training_frame(df)

    assert "home_form_points" in out.columns
    assert "away_form_points" in out.columns
    assert out.loc[1, "home_form_points"] == 3
    assert out.loc[2, "home_form_points"] == 6
