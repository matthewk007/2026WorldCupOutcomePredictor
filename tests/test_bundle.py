from pathlib import Path

import pandas as pd

from src.bundle import build_training_matches, load_kaggle_bundle, load_bundle_config


def test_load_bundle_config_reads_toml(tmp_path):
    config = tmp_path / "data_sources.toml"
    config.write_text('[files]\nmatches = "matches.csv"\nrankings = "rankings.csv"\n', encoding="utf-8")

    out = load_bundle_config(config)

    assert out["files"]["matches"] == "matches.csv"
    assert out["files"]["rankings"] == "rankings.csv"


def test_load_kaggle_bundle_loads_matches_and_rankings(tmp_path):
    data_dir = tmp_path
    pd.DataFrame(
        {
            "date": ["2018-06-14"],
            "home_team": ["Russia"],
            "away_team": ["Saudi Arabia"],
            "home_score": [5],
            "away_score": [0],
        }
    ).to_csv(data_dir / "matches.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2018-06-10", "2018-06-10"],
            "team": ["Russia", "Saudi Arabia"],
            "rank": [70, 67],
        }
    ).to_csv(data_dir / "rankings.csv", index=False)
    (data_dir / "data_sources.toml").write_text('[files]\nmatches = "matches.csv"\nrankings = "rankings.csv"\n', encoding="utf-8")

    bundle = load_kaggle_bundle(data_dir / "data_sources.toml")

    assert len(bundle.matches) == 1
    assert len(bundle.rankings) == 2


def test_build_training_matches_attaches_rankings(tmp_path):
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

    bundle = type("Bundle", (), {"matches": matches, "rankings": rankings})
    out = build_training_matches(bundle)

    assert out.loc[0, "home_rank"] == 70
    assert out.loc[0, "away_rank"] == 67
