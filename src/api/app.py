"""FastAPI inference service for both tasks.

Models are loaded once at import/startup (not per request). ``POST /predict`` takes
a typed request of *raw* game features plus a ``task`` discriminator and returns:

* ``task="outcome"`` -> predicted class + per-class probabilities;
* ``task="rating"``  -> predicted average rating.

Invalid input is rejected by pydantic with a 422 before any model runs. The request
intentionally exposes only the *legitimate* inputs per task (Task A is pre-game, so
no winner / move-count fields), mirroring the leakage framing in features.py.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import features as F

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Defaults for columns required by engineer() but irrelevant to a given task; they
# are dropped before the model sees them, so the value never affects a prediction.
_ENGINEER_DEFAULTS = {
    "white_rating": 1500, "black_rating": 1500, "turns": 60,
    "winner": "white", "victory_status": "resign",
}


@lru_cache(maxsize=1)
def load_models() -> dict[str, object]:
    """Load persisted pipelines once. Cached so it runs a single time per process."""
    paths = {"outcome": MODELS_DIR / "outcome_model.joblib",
             "rating": MODELS_DIR / "rating_model.joblib"}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Model artifact(s) not found: {missing}. Run `python -m src.train` first."
        )
    return {task: joblib.load(p) for task, p in paths.items()}


# --- request / response schemas --------------------------------------------
class GameFeatures(BaseModel):
    """Raw, pre-game-legitimate features. Ratings drive Task A; cadence/opening both."""
    rated: bool = Field(..., description="Whether the game was rated.")
    increment_code: str = Field(..., examples=["10+0"], description="Lichess 'base+inc'.")
    opening_eco: str = Field(..., examples=["C50"], description="ECO opening code.")
    opening_ply: int = Field(..., ge=0, le=200, description="Plies in the book opening.")

    # Task A (outcome) inputs — both players' ratings, known before the game.
    white_rating: int | None = Field(None, ge=100, le=4000)
    black_rating: int | None = Field(None, ge=100, le=4000)

    # Task B (rating) inputs — known about the completed game, fair for rating.
    turns: int | None = Field(None, ge=1, le=600, description="Move count (Task B only).")
    winner: Literal["white", "black", "draw"] | None = None
    victory_status: Literal["mate", "resign", "outoftime", "draw"] | None = None


class PredictRequest(BaseModel):
    task: Literal["outcome", "rating"]
    features: GameFeatures


class OutcomeResponse(BaseModel):
    task: Literal["outcome"] = "outcome"
    prediction: str
    probabilities: dict[str, float]


class RatingResponse(BaseModel):
    task: Literal["rating"] = "rating"
    prediction: float


_REQUIRED = {
    "outcome": ["white_rating", "black_rating"],
    "rating": ["turns", "winner", "victory_status"],
}


def _build_row(features: GameFeatures, task: str) -> pd.DataFrame:
    """Validate task-required fields and assemble a one-row frame for the pipeline."""
    data = features.model_dump()
    missing = [f for f in _REQUIRED[task] if data.get(f) is None]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"task='{task}' requires field(s): {missing}",
        )
    row = {**_ENGINEER_DEFAULTS, **{k: v for k, v in data.items() if v is not None}}
    return F.select_features(pd.DataFrame([row]), task)


app = FastAPI(title="Chess Elo Predictor", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=OutcomeResponse | RatingResponse)
def predict(req: PredictRequest):
    model = load_models()[req.task]
    X = _build_row(req.features, req.task)

    if req.task == "outcome":
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        return OutcomeResponse(
            prediction=str(model.predict(X)[0]),
            probabilities={c: float(p) for c, p in zip(classes, proba)},
        )
    return RatingResponse(prediction=float(model.predict(X)[0]))
