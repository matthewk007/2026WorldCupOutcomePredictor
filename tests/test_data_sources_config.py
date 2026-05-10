from src.bundle import load_bundle_config


def test_data_sources_config_points_at_root_csvs():
    config = load_bundle_config("data_sources.toml")

    assert config["files"]["matches"] == "matches_1930_2022.csv"
    assert config["files"]["rankings"] == "fifa_ranking_2022-10-06.csv"
