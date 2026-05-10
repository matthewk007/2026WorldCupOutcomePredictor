import pandas as pd

from src.bundle import latest_rank_snapshot, ranking_snapshot_before


def test_latest_rank_snapshot_handles_static_rank_file():
    rankings = pd.DataFrame(
        {
            "team": ["Russia", "Saudi Arabia"],
            "rank": [70, 67],
        }
    )

    snapshot = latest_rank_snapshot(rankings)

    assert snapshot["Russia"] == 70
    assert snapshot["Saudi Arabia"] == 67


def test_ranking_snapshot_before_falls_back_for_static_rank_file():
    rankings = pd.DataFrame(
        {
            "team": ["Russia", "Saudi Arabia"],
            "rank": [70, 67],
        }
    )

    snapshot = ranking_snapshot_before(rankings, "2026-06-01")

    assert snapshot["Russia"] == 70
    assert snapshot["Saudi Arabia"] == 67
