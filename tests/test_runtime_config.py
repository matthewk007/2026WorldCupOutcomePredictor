from src.predict import get_model_dir


def test_get_model_dir_uses_default(monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    assert str(get_model_dir("artifacts")) == "artifacts"


def test_get_model_dir_uses_environment(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "custom-artifacts")
    assert str(get_model_dir("artifacts")) == "custom-artifacts"
