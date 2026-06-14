# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repo is **pre-implementation**: it currently contains only `PRD_chess_elo_predictor.md`, the
authoritative spec. Build the project by working through the PRD's **§14 Build order**. The PRD is
the source of truth for scope, structure, and the definition of done — read it before acting.

## What this is

A small, honest ML project that predicts two things from features of a *completed* Lichess game:

- **Task A — Outcome** (classification): white win / black win / draw.
- **Task B — Rating** (regression): the game's average player rating.

This is descriptive prediction (post-game), **not** a live in-game predictor. The goal is
production-quality engineering for a portfolio (reproducible pipeline, baselines, real held-out
metrics, typed FastAPI service, Docker, tests), favoring clarity over cleverness.

## Tech stack & tooling

Python 3.12, managed with **uv**. pandas · scikit-learn · FastAPI · uvicorn · pydantic v2 ·
joblib · pytest · matplotlib (EDA only) · Docker. Keep dependencies minimal — ask before adding
anything heavyweight.

## Commands (target end state — wire these up as you build)

```bash
uv sync                                  # install deps
uv run python -m src.train               # train both tasks, eval vs baselines, write models/metrics.json
uv run pytest                            # full test suite
uv run pytest tests/test_features.py     # single test file (e.g. the leakage guard)
uv run pytest tests/test_features.py::test_no_leakage_columns   # single test
uv run uvicorn src.api.app:app --reload  # run API locally
docker build -t chess-elo .
docker run -p 8080:8080 chess-elo        # container must serve /health and /predict
```

The API binds to `$PORT` (default 8080) so it is Cloud-Run-ready unchanged.

## Architecture

Pipeline flows through `src/`: `data.py` (load + **inspect actual schema** + clean → persist to
`data/processed/`) → `features.py` (feature engineering + leakage guards) → `train.py` (baselines,
cross-validation, model selection, persist) using `evaluate.py` (metrics helpers) → `api/app.py`
(FastAPI loading the persisted model). Models and `metrics.json` land in `models/`.

Key cross-cutting design points:

- **Metrics are written to `models/metrics.json`, never restated by hand.** The README and tests
  read from that file. This is how the "never fabricate numbers" rule is enforced mechanically.
- **The model loads once at FastAPI startup**, not per request.
- `/predict` takes a typed pydantic request with a `task` field (`"outcome"` | `"rating"`):
  outcome returns predicted class + class probabilities; rating returns a value. Bad input → 422.

## Non-negotiable rules (from PRD §2, §5)

- **Never fabricate numbers.** Every metric in docs comes from an actual eval run. Unmeasured =
  a clearly-marked `TODO`, not a guess. An honest B+ result beats a suspiciously perfect one.
- **No data leakage (Task A).** Exclude any field encoding the result: `winner`, `victory_status`,
  and anything derived from them. `rating_diff` and `avg_rating` are legitimate (known pre-game).
  `move_count` is acceptable under the descriptive framing but must be noted as post-hoc in the
  README. **A unit test must assert the leakage columns are absent from the training matrix.**
- **Task B target leakage:** `avg_rating` is Task B's target — never use it as a Task B feature.
- **Target encoding** (e.g. for openings) must be done *inside* CV folds, not before splitting.
- **Reproducible:** fixed random seed; one command trains and evaluates end to end.
- **Beat a baseline:** mandatory baselines are majority-class + "higher-rated wins" (Task A) and
  predict-the-mean (Task B). If a model doesn't beat its baseline, report that honestly.

## Data

- Default dataset: Kaggle "Chess Game Dataset (Lichess)" (~20k games). Lichess open database / API
  are acceptable alternatives.
- **Inspect the real schema before writing feature code** — print columns, dtypes, missingness, a
  sample, and adapt. Do not hardcode column names from assumptions.
- Raw data is gitignored; document exactly how to obtain it in the README. Processed data persists
  to `data/processed/` for reproducibility.

## Conventions

- Prefer clear, typed, well-named code over abstraction — interviewers read this.
- Commit in logical chunks with meaningful messages; the git history is part of the story.
- `git init` has not been run yet (PRD build step 1). Out-of-scope for v1: cloud deploy, any
  LLM/GenAI, a frontend, the Stockfish `accuracy` feature, heavy hyperparameter tuning.
