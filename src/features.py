"""Feature engineering and leakage guards for both tasks.

Two kinds of transformation live here:

1. ``engineer`` — pure, **row-wise** derivations (parse the time control, derive
   ``rating_diff`` / ``avg_rating``, bucket the cadence). Because each value is a
   function of a single row, computing these before the train/test split is *not*
   leakage.

2. ``build_preprocessor`` — the part that must be *fitted* (one-hot encoding of
   openings/cadence). It is returned as an unfitted sklearn ``ColumnTransformer``
   so the training pipeline fits it on the training fold only. Infrequent openings
   collapse into a single "other" bucket via ``min_frequency``.

Leakage framing (PRD §5):
  * Task A (outcome): ``winner`` is the target; ``winner``, ``victory_status`` and
    the raw ``moves`` must never enter the feature matrix. ``assert_no_leakage``
    enforces this and is exercised by a unit test.
  * Task B (rating): ``avg_rating`` is the target; the ratings it is derived from
    (``white_rating``, ``black_rating``, ``rating_diff``) must not be features.
    ``winner`` and ``victory_status`` *are* legitimate inputs here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# --- Time-control buckets ---------------------------------------------------
# Lichess estimates a game's duration as base_seconds + 40 * increment_seconds
# and buckets it. We fold ultrabullet into bullet (negligible counts here).
BULLET_MAX = 180        # < 3 min
BLITZ_MAX = 480         # < 8 min
RAPID_MAX = 1500        # < 25 min; >= 1500 is classical

# Columns that encode the result of the game — forbidden as Task A features.
# `turns` (move count) is included: the side delivering mate / forcing resignation
# makes the last move, so the *parity* of the move count near-determines the winner
# (an odd ply count -> white won ~91% of the time in this data). The PRD allows
# move_count as a post-hoc feature, but that parity makes it effective leakage, so
# Task A is kept to genuine pre-game conditions. `turns` is still a fair Task B input.
LEAKAGE_COLS_OUTCOME = ["winner", "victory_status", "moves", "turns"]
# Columns the Task B target is derived from — forbidden as Task B features.
LEAKAGE_COLS_RATING = ["avg_rating", "white_rating", "black_rating", "rating_diff"]

# Feature groups per task (column names in the *engineered* frame).
OUTCOME_NUMERIC = [
    "white_rating", "black_rating", "rating_diff", "avg_rating",
    "base_seconds", "increment_seconds", "opening_ply",
]
OUTCOME_CATEGORICAL = ["rated", "time_control", "opening_eco"]

RATING_NUMERIC = ["turns", "base_seconds", "increment_seconds", "opening_ply"]
RATING_CATEGORICAL = ["rated", "time_control", "opening_eco", "winner", "victory_status"]

TARGET_OUTCOME = "winner"
TARGET_RATING = "avg_rating"


def _parse_time_control(increment_code: pd.Series) -> pd.DataFrame:
    """Parse "base+inc" (base in minutes, inc in seconds) into seconds + a bucket."""
    parts = increment_code.astype(str).str.split("+", n=1, expand=True)
    base_minutes = pd.to_numeric(parts[0], errors="coerce").fillna(0)
    increment_seconds = pd.to_numeric(parts[1], errors="coerce").fillna(0)
    base_seconds = base_minutes * 60
    estimated = base_seconds + 40 * increment_seconds

    bucket = np.select(
        [estimated < BULLET_MAX, estimated < BLITZ_MAX, estimated < RAPID_MAX],
        ["bullet", "blitz", "rapid"],
        default="classical",
    )
    return pd.DataFrame({
        "base_seconds": base_seconds.astype(int),
        "increment_seconds": increment_seconds.astype(int),
        "time_control": bucket,
    }, index=increment_code.index)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Pure and row-wise — safe to run before splitting."""
    out = df.copy()
    tc = _parse_time_control(out["increment_code"])
    out[["base_seconds", "increment_seconds", "time_control"]] = tc

    out["rating_diff"] = out["white_rating"] - out["black_rating"]
    out["avg_rating"] = (out["white_rating"] + out["black_rating"]) / 2.0
    return out


def assert_no_leakage(X: pd.DataFrame, task: str) -> None:
    """Raise if any forbidden column is present in the feature matrix for ``task``."""
    forbidden = LEAKAGE_COLS_OUTCOME if task == "outcome" else LEAKAGE_COLS_RATING
    present = [c for c in forbidden if c in X.columns]
    if present:
        raise AssertionError(
            f"Leakage: forbidden column(s) {present} present in Task '{task}' features."
        )


def get_xy(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for the given task from an engineered frame.

    Runs the leakage guard before returning, so it is impossible to obtain a
    feature matrix that contains a forbidden column.
    """
    eng = engineer(df)
    if task == "outcome":
        cols = OUTCOME_NUMERIC + OUTCOME_CATEGORICAL
        y = eng[TARGET_OUTCOME]
    elif task == "rating":
        cols = RATING_NUMERIC + RATING_CATEGORICAL
        y = eng[TARGET_RATING]
    else:
        raise ValueError(f"Unknown task {task!r}; expected 'outcome' or 'rating'.")

    X = eng[cols].copy()
    assert_no_leakage(X, task)
    return X, y


def feature_columns(task: str) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) for a task."""
    if task == "outcome":
        return list(OUTCOME_NUMERIC), list(OUTCOME_CATEGORICAL)
    if task == "rating":
        return list(RATING_NUMERIC), list(RATING_CATEGORICAL)
    raise ValueError(f"Unknown task {task!r}.")


def build_preprocessor(task: str) -> ColumnTransformer:
    """Unfitted preprocessor: passthrough numerics, one-hot categoricals.

    ``min_frequency`` collapses rare openings (and any rare cadence/status) into a
    single "infrequent" bucket — the PRD's "top-N openings + other" without manual
    top-N bookkeeping, and fold-safe because it is fitted inside the pipeline.
    """
    numeric, categorical = feature_columns(task)
    onehot = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=30,        # openings seen < 30 times collapse into "other"
        sparse_output=False,
    )
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", onehot, categorical),
        ],
        remainder="drop",
    )
