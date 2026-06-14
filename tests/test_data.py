"""Tests for cleaning behaviour in src.data."""

from src.data import DROP_COLS, clean


def test_clean_normalises_rated_to_bool(raw_like):
    out = clean(raw_like)
    assert out["rated"].dtype == bool
    assert set(out["rated"].unique()) <= {True, False}


def test_clean_drops_identifier_and_timestamp_columns(raw_like):
    out = clean(raw_like)
    for col in DROP_COLS:
        assert col not in out.columns


def test_clean_removes_zero_turn_and_duplicate_rows(raw_like):
    out = clean(raw_like)
    # 4 raw rows: one zero-turn ('b') and its duplicate are both removed -> 2 remain.
    assert (out["turns"] > 0).all()
    assert len(out) == 2


def test_clean_enforces_integer_dtypes(raw_like):
    out = clean(raw_like)
    for col in ["white_rating", "black_rating", "turns", "opening_ply"]:
        assert out[col].dtype.kind == "i"
