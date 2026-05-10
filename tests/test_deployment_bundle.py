import pandas as pd

from src.model import ensure_artifacts_from_bundle


def test_ensure_artifacts_from_bundle_trains_from_config(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    pd.DataFrame(
        {
            "Date": ["2018-06-14", "2018-06-15"],
            "home_team": ["Russia", "Egypt"],
            "away_team": ["Saudi Arabia", "Uruguay"],
            "home_score": [5, 1],
            "away_score": [0, 0],
        }
    ).to_csv(data_dir / "matches.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2018-06-10", "2018-06-10"],
            "team": ["Russia", "Saudi Arabia"],
            "rank": [70, 67],
        }
    ).to_csv(data_dir / "rankings.csv", index=False)
    (tmp_path / "data_sources.toml").write_text('[files]\nmatches = "data/matches.csv"\nrankings = "data/rankings.csv"\n', encoding="utf-8")

    artifacts = ensure_artifacts_from_bundle(tmp_path / "artifacts", tmp_path / "data_sources.toml")

    assert artifacts.classifier_path.exists()
    assert artifacts.regressor_path.exists()
