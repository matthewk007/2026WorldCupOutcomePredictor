import pandas as pd

from src.data import load_matches


def test_load_matches_returns_dataframe(tmp_path):
    csv_path = tmp_path / "matches.csv"
    pd.DataFrame(
        {
            "date": ["2018-06-14"],
            "home_team": ["Russia"],
            "away_team": ["Saudi Arabia"],
            "home_score": [5],
            "away_score": [0],
        }
    ).to_csv(csv_path, index=False)

    df = load_matches(csv_path)
    assert list(df.columns)[:4] == ["date", "home_team", "away_team", "home_score"]
    assert len(df) == 1
