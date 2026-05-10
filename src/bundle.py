from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

import pandas as pd

from src.data import load_matches, normalize_match_columns
from src.features import attach_rankings


@dataclass
class KaggleBundle:
    matches: pd.DataFrame
    rankings: pd.DataFrame | None = None


def load_bundle_config(path) -> dict:
    config_path = Path(path)
    with config_path.open("rb") as fh:
        return tomllib.load(fh)


def load_kaggle_bundle(config_path) -> KaggleBundle:
    config = load_bundle_config(config_path)
    base_dir = Path(config_path).parent
    files = config.get("files", {})

    matches_path = base_dir / files["matches"]
    matches = load_matches(matches_path)

    rankings = None
    rankings_file = files.get("rankings")
    if rankings_file:
        rankings = normalize_match_columns(pd.read_csv(base_dir / rankings_file))
        if "date" in rankings.columns:
            rankings["date"] = pd.to_datetime(rankings["date"], errors="coerce")
            rankings = rankings.dropna(subset=["date", "team", "rank"])
        else:
            rankings = rankings.dropna(subset=["team", "rank"])
        rankings["rank"] = rankings["rank"].astype(int)

    return KaggleBundle(matches=matches, rankings=rankings)


def build_training_matches(bundle: KaggleBundle) -> pd.DataFrame:
    matches = bundle.matches
    if bundle.rankings is not None and {"team", "rank", "date"}.issubset(bundle.rankings.columns):
        matches = attach_rankings(matches, bundle.rankings)
    return matches


def latest_rank_snapshot(rankings: pd.DataFrame) -> dict[str, int]:
    rankings_out = rankings.copy()
    if "date" in rankings_out.columns:
        rankings_out = rankings_out.sort_values("date")
    latest = rankings_out.groupby("team", as_index=False).tail(1)
    return {row["team"]: int(row["rank"]) for _, row in latest.iterrows()}


def ranking_snapshot_before(rankings: pd.DataFrame, cutoff_date) -> dict[str, int]:
    rankings_out = rankings.copy()
    if "date" not in rankings_out.columns:
        return latest_rank_snapshot(rankings_out)
    rankings_out = rankings_out[rankings_out["date"] < pd.to_datetime(cutoff_date)]
    if rankings_out.empty:
        return {}
    return latest_rank_snapshot(rankings_out)
