import pandas as pd

from src.data import load_matches, normalize_match_columns


def test_normalize_match_columns_handles_actual_kaggle_headers():
    df = pd.DataFrame(
        {
            "Date": ["2018-06-14"],
            "home_team": ["Russia"],
            "away_team": ["Saudi Arabia"],
            "home_score": [5],
            "away_score": [0],
            "Venue": ["Luzhniki Stadium"],
        }
    )

    out = normalize_match_columns(df)

    assert "date" in out.columns
    assert "venue" in out.columns


def test_load_matches_accepts_actual_kaggle_style_columns(tmp_path):
    csv_path = tmp_path / "matches_1930_2022.csv"
    pd.DataFrame(
        {
            "Date": ["2018-06-14"],
            "home_team": ["Russia"],
            "away_team": ["Saudi Arabia"],
            "home_score": [5],
            "away_score": [0],
            "Venue": ["Luzhniki Stadium"],
        }
    ).to_csv(csv_path, index=False)

    df = load_matches(csv_path)

    assert list(df.columns)[:5] == ["date", "home_team", "away_team", "home_score", "away_score"]
    assert df.loc[0, "venue"] == "Luzhniki Stadium"
