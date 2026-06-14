"""Load, inspect, and clean the raw Lichess games dataset.

The raw file is the classic Kaggle "Chess Game Dataset (Lichess)" (datasnaek),
~20k games. We never hardcode assumptions about the schema blindly: ``inspect``
prints columns/dtypes/missingness so the schema can be verified, and ``clean``
performs only the minimal, documented normalisations the real data needs.

Run ``python -m src.data`` to (re)build ``data/processed/games_clean.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Project paths (this file lives in src/).
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "games.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "games_clean.parquet"

# Columns dropped during cleaning: identifiers and raw timestamps that carry no
# generalisable signal for either task (player ids would memorise; timestamps are
# game metadata). The move list is kept out of the feature matrix but retained in
# the processed file in case future work wants engine analysis (PRD §6 stretch).
DROP_COLS = ["id", "white_id", "black_id", "created_at", "last_move_at"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV. Raises a clear error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. See the README for how to obtain it "
            "(Kaggle 'Chess Game Dataset (Lichess)') and place it at data/raw/games.csv."
        )
    return pd.read_csv(path)


def inspect(df: pd.DataFrame) -> None:
    """Print schema, dtypes, missingness and a sample — verify before coding features."""
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print("\ndtypes:")
    print(df.dtypes.to_string())
    print("\nmissing per column:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "  (none)")
    print("\nsample row:")
    print(df.iloc[0].to_string())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the raw frame into a tidy, typed table.

    Normalisations (each addresses a quirk verified in the real data):
      * ``rated`` arrives mixed-case ("True"/"TRUE"/"False"/"FALSE") -> bool.
      * Drop identifier and timestamp columns (see ``DROP_COLS``).
      * Ensure integer dtypes for ratings and turns.
      * Drop the handful of degenerate games with zero turns (no game was played).
      * De-duplicate. After ids/timestamps are dropped, rows that are identical on
        every remaining column (ratings, full move list, opening, result, ...) are
        treated as the same game: ~429 are exact duplicate records (same game id)
        and the rest share an identical move sequence and ratings. Dropping them
        keeps an identical game from landing in both the train and test splits,
        which would make held-out metrics optimistic.

    Leakage columns (``winner``, ``victory_status``) are intentionally retained
    here — they are the Task A target / Task B inputs. The leakage guard that
    excludes them from Task A's feature matrix lives in ``features.py``.
    """
    df = df.copy()

    df["rated"] = (
        df["rated"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
    )
    if df["rated"].isna().any():
        raise ValueError("Unexpected values in 'rated' column after normalisation.")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    for col in ["white_rating", "black_rating", "turns", "opening_ply"]:
        df[col] = df[col].astype(int)

    df = df[df["turns"] > 0]
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_processed(raw_path: Path = RAW_PATH, out_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Full pipeline: load -> inspect -> clean -> persist to parquet."""
    raw = load_raw(raw_path)
    print("=== raw ===")
    inspect(raw)

    clean_df = clean(raw)
    print("\n=== cleaned ===")
    print(f"shape: {clean_df.shape[0]} rows x {clean_df.shape[1]} cols "
          f"({raw.shape[0] - clean_df.shape[0]} rows removed)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path.relative_to(ROOT)}")
    return clean_df


def load_processed(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load the cleaned dataset, building it from raw if it does not yet exist."""
    if not path.exists():
        return build_processed(out_path=path)
    return pd.read_parquet(path)


if __name__ == "__main__":
    build_processed()
