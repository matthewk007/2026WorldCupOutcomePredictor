import pandas as pd

from src.features import attach_rankings


def test_attach_rankings_merges_home_and_away_rankings():
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-06-14"]),
            "home_team": ["Russia"],
            "away_team": ["Saudi Arabia"],
            "home_score": [5],
            "away_score": [0],
        }
    )
    rankings = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-06-10", "2018-06-10"]),
            "team": ["Russia", "Saudi Arabia"],
            "rank": [70, 67],
        }
    )

    out = attach_rankings(matches, rankings)

    assert out.loc[0, "home_rank"] == 70
    assert out.loc[0, "away_rank"] == 67
