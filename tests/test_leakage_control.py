import pandas as pd

from src.bundle import latest_rank_snapshot
from src.features import make_inference_row
from src.train import time_based_train_test_split


def test_time_based_train_test_split_keeps_chronology():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-06-01", "2018-06-02", "2018-06-03", "2018-06-04", "2018-06-05"]),
            "home_team": ["A", "B", "C", "D", "E"],
            "away_team": ["F", "G", "H", "I", "J"],
            "home_score": [1, 1, 1, 1, 1],
            "away_score": [0, 0, 0, 0, 0],
        }
    )

    train_df, test_df = time_based_train_test_split(frame, test_size=0.4)

    assert train_df["date"].max() < test_df["date"].min()


def test_latest_rank_snapshot_uses_most_recent_rank_per_team():
    rankings = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-06-01", "2018-07-01", "2018-06-01"]),
            "team": ["Russia", "Russia", "Saudi Arabia"],
            "rank": [70, 65, 67],
        }
    )

    snapshot = latest_rank_snapshot(rankings)

    assert snapshot["Russia"] == 65
    assert snapshot["Saudi Arabia"] == 67


def test_make_inference_row_uses_rank_snapshot():
    row = make_inference_row(
        "Russia",
        "Saudi Arabia",
        match_date="2026-06-01",
        tournament="World Cup",
        ranking_snapshot={"Russia": 65, "Saudi Arabia": 67},
    )

    assert row.loc[0, "home_rank"] == 65
    assert row.loc[0, "away_rank"] == 67
