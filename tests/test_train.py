import pandas as pd

from src.model import train_models


def test_train_models_returns_artifacts(tmp_path):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-06-14", "2018-06-15", "2018-06-16", "2018-06-17"]),
            "home_team": ["Russia", "Egypt", "Brazil", "Germany"],
            "away_team": ["Saudi Arabia", "Uruguay", "Switzerland", "Mexico"],
            "home_score": [5, 1, 1, 0],
            "away_score": [0, 0, 1, 1],
        }
    )

    artifacts = train_models(df, output_dir=tmp_path)
    assert artifacts.classifier_path.exists()
    assert artifacts.regressor_path.exists()
