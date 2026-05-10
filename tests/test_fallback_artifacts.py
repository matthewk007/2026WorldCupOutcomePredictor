from src.model import ensure_artifacts


def test_ensure_artifacts_creates_missing_files(tmp_path):
    artifacts = ensure_artifacts(tmp_path)

    assert artifacts.classifier_path.exists()
    assert artifacts.regressor_path.exists()
