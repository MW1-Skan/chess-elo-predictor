"""API tests (PRD §10). Use the persisted model artifacts in models/."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)

OUTCOME_REQ = {
    "task": "outcome",
    "features": {
        "rated": True, "increment_code": "10+0", "opening_eco": "C50",
        "opening_ply": 5, "white_rating": 1800, "black_rating": 1500,
    },
}
RATING_REQ = {
    "task": "rating",
    "features": {
        "rated": True, "increment_code": "15+15", "opening_eco": "B01",
        "opening_ply": 4, "turns": 55, "winner": "black", "victory_status": "resign",
    },
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Chess Elo Predictor" in r.text


def test_predict_outcome_shape():
    r = client.post("/predict", json=OUTCOME_REQ)
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "outcome"
    assert body["prediction"] in {"white", "black", "draw"}
    probs = body["probabilities"]
    assert set(probs) == {"white", "black", "draw"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_rating_shape():
    r = client.post("/predict", json=RATING_REQ)
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "rating"
    assert isinstance(body["prediction"], float)
    assert 500 < body["prediction"] < 3000


def test_predict_missing_task_required_field_returns_422():
    bad = {"task": "outcome", "features": {
        "rated": True, "increment_code": "10+0", "opening_eco": "C50", "opening_ply": 5,
    }}  # no ratings
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_bad_enum_returns_422():
    bad = {"task": "rating", "features": {**RATING_REQ["features"], "winner": "purple"}}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_bad_rating_range_returns_422():
    bad = {"task": "outcome", "features": {**OUTCOME_REQ["features"], "white_rating": 99}}
    assert client.post("/predict", json=bad).status_code == 422
