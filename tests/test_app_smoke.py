import importlib

import pytest


def test_fixture_loader_fails_without_official_file(tmp_path, monkeypatch):
    import app

    missing = tmp_path / "official_2026_fixtures.csv"
    monkeypatch.setattr(app, "DEFAULT_FIXTURE_PATH", missing)

    with pytest.raises(FileNotFoundError):
        app.main()


def test_app_module_imports():
    importlib.import_module("app")
