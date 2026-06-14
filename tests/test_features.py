"""Feature-engineering and leakage-guard tests (PRD §5)."""

import pytest

from src import features as F


def test_engineer_derives_expected_columns(signal_df):
    eng = F.engineer(signal_df)
    for col in ["base_seconds", "increment_seconds", "time_control",
                "rating_diff", "avg_rating"]:
        assert col in eng.columns
    # 10+0 -> 600s estimated -> rapid; 15+15 -> 1500s -> classical.
    tc = F.engineer(signal_df.assign(increment_code="10+0"))["time_control"]
    assert (tc == "rapid").all()
    tc2 = F.engineer(signal_df.assign(increment_code="15+15"))["time_control"]
    assert (tc2 == "classical").all()


@pytest.mark.parametrize("forbidden", ["winner", "victory_status", "moves", "turns"])
def test_outcome_feature_matrix_excludes_leakage_columns(signal_df, forbidden):
    """The core leakage guard: result-encoding columns must be absent from Task A X."""
    X = F.select_features(signal_df, "outcome")
    assert forbidden not in X.columns


def test_rating_feature_matrix_excludes_target_components(signal_df):
    X = F.select_features(signal_df, "rating")
    for forbidden in F.LEAKAGE_COLS_RATING:  # avg_rating, white/black_rating, rating_diff
        assert forbidden not in X.columns
    # winner/victory_status ARE legitimate Task B inputs.
    assert "winner" in X.columns and "victory_status" in X.columns


def test_assert_no_leakage_raises_on_forbidden_column(signal_df):
    X = F.select_features(signal_df, "outcome").assign(winner="white")
    with pytest.raises(AssertionError):
        F.assert_no_leakage(X, "outcome")


def test_get_xy_targets(signal_df):
    _, y_out = F.get_xy(signal_df, "outcome")
    assert set(y_out.unique()) <= {"white", "black", "draw"}
    _, y_rat = F.get_xy(signal_df, "rating")
    assert y_rat.dtype.kind == "f"
