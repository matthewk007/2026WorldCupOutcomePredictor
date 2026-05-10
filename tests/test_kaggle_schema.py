import pandas as pd

from src.data import normalize_match_columns


def test_normalize_match_columns_maps_common_kaggle_names():
    df = pd.DataFrame(
        {
            "date": ["2018-06-14"],
            "home_team": ["Russia"],
            "away_team": ["Saudi Arabia"],
            "home_score": [5],
            "away_score": [0],
            "tournament": ["World Cup"],
        }
    )

    out = normalize_match_columns(df)

    assert list(out.columns)[:5] == ["date", "home_team", "away_team", "home_score", "away_score"]
    assert out.loc[0, "tournament"] == "World Cup"
