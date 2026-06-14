"""Sanity test: trained models beat the trivial baselines on a synthetic fixture."""

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

from src import features as F
from src.train import build_models


def test_outcome_model_beats_majority_baseline(signal_df):
    X, y = F.get_xy(signal_df, "outcome")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    model = build_models("outcome")["tree"].fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))

    baseline = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    base_acc = accuracy_score(yte, baseline.predict(Xte))

    assert acc > base_acc


def test_rating_model_beats_mean_baseline(signal_df):
    X, y = F.get_xy(signal_df, "rating")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

    model = build_models("rating")["tree"].fit(Xtr, ytr)
    mae = mean_absolute_error(yte, model.predict(Xte))

    baseline = DummyRegressor(strategy="mean").fit(Xtr, ytr)
    base_mae = mean_absolute_error(yte, baseline.predict(Xte))

    assert mae < base_mae
