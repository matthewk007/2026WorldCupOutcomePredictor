from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score"]

COLUMN_ALIASES = {
    "Date": "date",
    "city": "venue",
    "Venue": "venue",
    "Officials": "officials",
    "Round": "round",
    "Host": "host",
    "Year": "year",
    "home_team_name": "home_team",
    "away_team_name": "away_team",
    "country": "team",
    "team_name": "team",
    "ranking": "rank",
    "home_score": "home_score",
    "away_score": "away_score",
    "tournament": "tournament",
}


def normalize_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    renamed = {column: COLUMN_ALIASES.get(column, column) for column in out.columns}
    out = out.rename(columns=renamed)
    return out


def load_matches(path):
    df = normalize_match_columns(pd.read_csv(Path(path)))
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)
