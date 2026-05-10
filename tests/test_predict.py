from src.predict import format_prediction


def test_format_prediction_structure():
    result = format_prediction(
        outcome_probs={"home_win": 0.6, "draw": 0.2, "away_win": 0.2},
        expected_score=(2, 1),
    )
    assert result["predicted_outcome"] == "home_win"
    assert result["expected_score"] == "2-1"
