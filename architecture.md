# Architecture

This document explains *how* the project is put together and *why* the key
engineering choices were made. For results and run instructions see `README.md`.

## Pipeline overview

The system is a linear data pipeline with two consumers (training and serving) that
share the same feature code:

```
data/raw/games.csv
      │  src/data.py        load → inspect schema → clean → dedup
      ▼
data/processed/games_clean.parquet
      │  src/features.py    engineer() (row-wise)  +  build_preprocessor() (fit-time)
      ▼
   feature matrix X, target y     ──────────────┐
      │  src/train.py                            │  src/api/app.py
      ▼                                          ▼
 models/outcome_model.joblib              load pipelines once,
 models/rating_model.joblib               serve /predict
 models/metrics.json
```

Two design rules drive the structure:

1. **One source of feature truth.** Training and the API both call
   `src.features`, so the transformations applied at inference are byte-for-byte the
   ones used in training. The persisted artifact is a full sklearn `Pipeline`
   (preprocessing + estimator), so the API never re-implements preprocessing.
2. **Numbers live in one file.** Every metric is written to `models/metrics.json`
   by `train.py`; the README and tests read it. Nothing is transcribed by hand,
   which is how the "never fabricate numbers" rule is enforced mechanically.

## Module responsibilities

| Module | Responsibility | Notable choices |
|---|---|---|
| `src/data.py` | Load raw CSV, print schema/missingness, clean, persist parquet | Normalises mixed-case `rated`; drops ids/timestamps; conservative dedup |
| `src/features.py` | Feature engineering, leakage guards, preprocessor factory | Splits *row-wise* derivations from *fit-time* encoding |
| `src/evaluate.py` | Metric helpers returning JSON-able dicts | Plus the "higher-rated wins" heuristic baseline |
| `src/train.py` | Split, CV model selection, baselines, importance, persist | Held-out test scored exactly once |
| `src/api/app.py` | FastAPI service | Models loaded once via `lru_cache`; typed pydantic I/O |

## Leakage strategy (the core of the project)

Leakage is handled in three layers rather than relying on discipline:

1. **Explicit forbidden-column lists** per task in `features.py`
   (`LEAKAGE_COLS_OUTCOME`, `LEAKAGE_COLS_RATING`).
2. **An assertion in the data path.** `assert_no_leakage` runs inside
   `select_features`/`get_xy`, so it is impossible to obtain a feature matrix that
   contains a forbidden column — at train *or* inference time.
3. **A parametrized unit test** that fails if any forbidden column reappears.

Two leaks are guarded specifically:

- **Result-encoding columns** (`winner`, `victory_status`, `moves`) — obvious leaks
  for Task A.
- **Move-count parity** (`turns`) — a *subtle* leak. The player who ends the game
  makes the last move, so move-count parity nearly determines the winner. `turns` is
  therefore excluded from Task A (but kept for Task B, where it legitimately signals
  player strength). This was discovered during EDA and is demonstrated in the
  notebook; it is the most important judgment call in the project.

### Why preprocessing lives inside the Pipeline

Openings are high-cardinality (365 ECO codes / 1477 names). They are encoded with
`OneHotEncoder(min_frequency=30)` **inside** the sklearn `Pipeline`, so:

- the encoder is fit on the **training fold only** during cross-validation (no
  category/frequency information leaks across the split);
- rare openings collapse into a single "infrequent" bucket — the PRD's "top-N
  openings + other" without manual bookkeeping;
- unseen openings at inference map to that bucket via
  `handle_unknown="infrequent_if_exist"`.

Row-wise features (`rating_diff`, `avg_rating`, time-control buckets) are *pure
functions of a single row* and carry no cross-row state, so computing them before
the split is **not** leakage; they live in `engineer()` outside the fitted pipeline.

## Modelling protocol

- **Split:** 80/20, stratified on outcome for Task A, fixed seed 42.
- **Selection:** 5-fold CV on the training set compares a linear model
  (LogisticRegression / Ridge, with standardisation) against a gradient-boosted tree
  (HistGradientBoosting). The CV-best model is refit on the full training set and
  scored **once** on the held-out test set.
- **Baselines** are computed on the same test split so the comparison is apples to
  apples.

For Task A the linear model wins CV by a hair and is selected; for Task B the tree
wins on MAE. Both outcomes are recorded in `metrics.json` under each task's `cv` key.

## Serving design

- The FastAPI app loads both persisted pipelines once (`lru_cache`) — not per
  request.
- The request schema (`GameFeatures`) exposes **only the legitimate inputs**: Task A
  callers supply ratings; Task B callers supply move count / result fields. A
  validator enforces the task-required subset and returns `422` if it is missing;
  pydantic enforces types/enums/ranges and returns `422` automatically.
- The app resolves `models/` relative to the repo root and is run from source
  (`PYTHONPATH=/app`) in Docker, so the same model paths work locally and in the
  container.

## Reproducibility & packaging

- **uv** pins Python 3.12 and a locked dependency set (`uv.lock`); the same lock is
  used locally and in the Docker build.
- A single command (`python -m src.train`) rebuilds processed data if needed, trains
  both tasks, and rewrites all artifacts deterministically (seed 42).
- The Docker image installs runtime dependencies only and bakes in the persisted
  models; it binds to `$PORT` (default 8080) for zero-change Cloud Run deployment.
