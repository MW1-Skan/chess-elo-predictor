"""Train, cross-validate, evaluate against baselines, and persist both models.

One command does everything reproducibly::

    python -m src.train

It writes ``models/outcome_model.joblib``, ``models/rating_model.joblib`` and
``models/metrics.json`` (every number the README/tests quote comes from here).

Protocol (PRD §7): an 80/20 train/test split (stratified on the outcome for
Task A); model selection by 5-fold CV on the *training* set only; the held-out
test set is touched exactly once, for the final reported numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import evaluate as E
from src import features as F
from src.data import ROOT, load_processed

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MODELS_DIR = ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"


# --- model definitions ------------------------------------------------------
def build_models(task: str) -> dict[str, Pipeline]:
    """Two pipelines per task: a linear model and a gradient-boosted tree.

    The linear model standardises features after one-hot encoding; the tree does
    not need scaling. Both share the same fold-safe preprocessor from features.py.
    """
    if task == "outcome":
        linear = LogisticRegression(max_iter=2000, random_state=SEED)
        tree = HistGradientBoostingClassifier(random_state=SEED)
    else:
        linear = Ridge(random_state=SEED)
        tree = HistGradientBoostingRegressor(random_state=SEED)

    return {
        "linear": Pipeline([
            ("pre", F.build_preprocessor(task)),
            ("scale", StandardScaler()),
            ("model", linear),
        ]),
        "tree": Pipeline([
            ("pre", F.build_preprocessor(task)),
            ("model", tree),
        ]),
    }


def stratified_split(X: pd.DataFrame, y: pd.Series, stratify: pd.Series | None):
    """Deterministic 80/20 split."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=stratify
    )


def _permutation_importance(model, X_test, y_test, scoring) -> list[dict]:
    """Permutation importance per *raw* input column (averaged), most important first."""
    result = permutation_importance(
        model, X_test, y_test, scoring=scoring,
        n_repeats=5, random_state=SEED,
    )
    ranked = sorted(
        ({"feature": col, "importance": float(m), "std": float(s)}
         for col, m, s in zip(X_test.columns, result.importances_mean,
                              result.importances_std)),
        key=lambda d: d["importance"], reverse=True,
    )
    return ranked


# --- Task A: outcome --------------------------------------------------------
def train_outcome(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60 + "\nTASK A — outcome (classification)\n" + "=" * 60)
    X, y = F.get_xy(df, "outcome")
    # Use the classifier's own (lexicographic) class order everywhere so that the
    # confusion matrix and the predict_proba columns line up — required for a
    # correct log-loss.
    labels = ["black", "draw", "white"]
    X_train, X_test, y_train, y_test = stratified_split(X, y, stratify=y)

    # Baselines (evaluated on the held-out test set).
    majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base_majority = E.outcome_metrics(y_test, majority.predict(X_test), labels=labels)
    base_heuristic = E.outcome_metrics(
        y_test, E.higher_rated_wins(X_test["rating_diff"]), labels=labels
    )
    print(f"baseline majority-class : acc={base_majority['accuracy']:.3f} "
          f"macroF1={base_majority['macro_f1']:.3f}")
    print(f"baseline higher-rated   : acc={base_heuristic['accuracy']:.3f} "
          f"macroF1={base_heuristic['macro_f1']:.3f}")

    # Cross-validated model selection on the training set.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scoring = {"accuracy": "accuracy", "macro_f1": "f1_macro", "neg_log_loss": "neg_log_loss"}
    cv_report = {}
    for name, pipe in build_models("outcome").items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        cv_report[name] = {
            "accuracy_mean": float(scores["test_accuracy"].mean()),
            "macro_f1_mean": float(scores["test_macro_f1"].mean()),
            "macro_f1_std": float(scores["test_macro_f1"].std()),
            "log_loss_mean": float(-scores["test_neg_log_loss"].mean()),
        }
        print(f"CV {name:7s}: acc={cv_report[name]['accuracy_mean']:.3f} "
              f"macroF1={cv_report[name]['macro_f1_mean']:.3f} "
              f"logloss={cv_report[name]['log_loss_mean']:.3f}")

    selected = max(cv_report, key=lambda n: cv_report[n]["macro_f1_mean"])
    print(f"selected: {selected}")

    # Refit selected model on full training set; evaluate once on test.
    best = build_models("outcome")[selected].fit(X_train, y_train)
    proba = best.predict_proba(X_test)
    test = E.outcome_metrics(y_test, best.predict(X_test), y_proba=proba, labels=labels)
    test["proba_columns"] = list(best.classes_)
    print(f"TEST {selected}: acc={test['accuracy']:.3f} macroF1={test['macro_f1']:.3f} "
          f"logloss={test['log_loss']:.3f}")

    importance = _permutation_importance(best, X_test, y_test, scoring="f1_macro")

    joblib.dump(best, MODELS_DIR / "outcome_model.joblib")
    return {
        "baselines": {"majority_class": base_majority, "higher_rated_wins": base_heuristic},
        "cv": cv_report,
        "selected_model": selected,
        "test": test,
        "feature_importance": importance,
    }


# --- Task B: rating ---------------------------------------------------------
def train_rating(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60 + "\nTASK B — rating (regression)\n" + "=" * 60)
    X, y = F.get_xy(df, "rating")
    X_train, X_test, y_train, y_test = stratified_split(X, y, stratify=None)

    mean_base = DummyRegressor(strategy="mean").fit(X_train, y_train)
    base_mean = E.rating_metrics(y_test, mean_base.predict(X_test))
    print(f"baseline predict-mean : MAE={base_mean['mae']:.1f} "
          f"RMSE={base_mean['rmse']:.1f} R2={base_mean['r2']:.3f}")

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scoring = {"neg_mae": "neg_mean_absolute_error", "r2": "r2"}
    cv_report = {}
    for name, pipe in build_models("rating").items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        cv_report[name] = {
            "mae_mean": float(-scores["test_neg_mae"].mean()),
            "mae_std": float(scores["test_neg_mae"].std()),
            "r2_mean": float(scores["test_r2"].mean()),
        }
        print(f"CV {name:7s}: MAE={cv_report[name]['mae_mean']:.1f} "
              f"R2={cv_report[name]['r2_mean']:.3f}")

    selected = min(cv_report, key=lambda n: cv_report[n]["mae_mean"])
    print(f"selected: {selected}")

    best = build_models("rating")[selected].fit(X_train, y_train)
    test = E.rating_metrics(y_test, best.predict(X_test))
    print(f"TEST {selected}: MAE={test['mae']:.1f} RMSE={test['rmse']:.1f} R2={test['r2']:.3f}")

    importance = _permutation_importance(
        best, X_test, y_test, scoring="neg_mean_absolute_error"
    )

    joblib.dump(best, MODELS_DIR / "rating_model.joblib")
    return {
        "baselines": {"predict_mean": base_mean},
        "cv": cv_report,
        "selected_model": selected,
        "test": test,
        "feature_importance": importance,
    }


def main() -> None:
    np.random.seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_processed()
    print(f"loaded {len(df)} games")

    metrics = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "n_rows": int(len(df)),
        "outcome": train_outcome(df),
        "rating": train_rating(df),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {METRICS_PATH.relative_to(ROOT)} and model artifacts to models/")


if __name__ == "__main__":
    main()
