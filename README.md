# Chess Game-Outcome & Elo Predictor

A small, honest, end-to-end ML project that predicts two things from features of a
**completed** Lichess game:

- **Task A — Outcome** (classification): white win / black win / draw.
- **Task B — Rating** (regression): the average player rating of the game.

It is built as production-style engineering rather than a notebook dump: a
reproducible pipeline, mandatory baselines, real held-out metrics, explicit
leakage guards, a typed FastAPI service, Docker packaging, and tests.

> **All numbers below come from an actual run** (`models/metrics.json`, seed 42).
> Re-running `python -m src.train` reproduces them.

---

## Overview

This is **descriptive** prediction, not a live in-game predictor: given the
conditions of a finished game, what was the likely result / how strong were the
players? Task A deliberately uses **pre-game information only** (ratings, opening,
time control) — see [Key design decisions](#key-design-decisions).

## Architecture

```
raw games.csv ──▶ src/data.py ──▶ data/processed/games_clean.parquet
                  (load, inspect schema, clean, dedup)
                                        │
                                        ▼
                  src/features.py  (engineer + leakage guards)
                   • row-wise: time-control buckets, rating_diff, avg_rating
                   • fold-safe preprocessor: one-hot openings (rare → "other")
                                        │
              ┌─────────────────────────┴─────────────────────────┐
              ▼                                                     ▼
        src/train.py                                         src/api/app.py
   80/20 split, 5-fold CV model                       loads persisted pipelines
   selection (linear vs HGB),                         once at startup; typed
   baselines, permutation importance                  /predict (outcome | rating)
              │                                                     │
              ▼                                                     ▼
   models/*.joblib + metrics.json  ───────────────────────▶  Dockerfile ($PORT)
```

`src/evaluate.py` holds the metric helpers; everything that produces a number
writes it to `models/metrics.json` so docs and tests read it rather than restating
it by hand.

## Key design decisions

- **Task A uses pre-game features only — `move_count` is excluded.** The side that
  delivers mate / forces resignation makes the *last* move, so the **parity** of the
  move count nearly determines the winner (odd ply count → white won ~91%; a
  parity-only rule scores ~0.90 on decisive games). Including `move_count` inflated
  accuracy to ~0.83 through this artifact rather than real skill, so it is treated as
  leakage for Task A and enforced absent by a unit test. It remains a legitimate
  **Task B** input.
- **Leakage guards are code, not comments.** `winner`, `victory_status`, `moves`,
  `turns` are forbidden in Task A's matrix; the rating components
  (`white_rating`, `black_rating`, `rating_diff`, `avg_rating`) are forbidden in Task
  B's. `assert_no_leakage` runs inside `get_xy`/`select_features` and is covered by a
  parametrized test.
- **Fold-safe encoding.** Openings are one-hot encoded inside the sklearn pipeline
  (`OneHotEncoder(min_frequency=…)`), so rare openings collapse to "other" and the
  encoder is fit on the training fold only — no target/category leakage across the
  split.
- **Conservative de-duplication.** Rows identical after dropping ids/timestamps (same
  ratings + full move list) are dropped so an identical game cannot span train and
  test, which would make held-out metrics optimistic (20,058 → 19,112 rows).

## Results & metrics

Held-out **test set** (20% of 19,112 games), metrics from `models/metrics.json`.

### Task A — Outcome (3-class)

| Model | Accuracy | Macro-F1 | Log-loss |
|---|---|---|---|
| Baseline: majority class | 0.499 | 0.222 | — |
| Baseline: higher-rated wins | 0.621 | 0.423 | — |
| **Selected: Logistic Regression** | **0.619** | **0.421** | **0.773** |

**Honest takeaway:** with pre-game features only, the model **beats the trivial
majority-class baseline but essentially ties the strong "higher-rated wins"
heuristic** — it does not beat it. `rating_diff` is the dominant (nearly the only)
signal; draws (~5% of games) are almost never predicted. This is a genuinely hard
problem from pre-game information, and the result is reported as-is.

### Task B — Rating (regression)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline: predict mean | 210.6 | 261.9 | 0.000 |
| **Selected: HistGradientBoosting** | **179.0** | **225.7** | **0.257** |

**Honest takeaway:** the model **clearly beats** the mean baseline (MAE −32 Elo,
R² 0.26). Top features (permutation importance): `opening_ply`, `opening_eco`,
`victory_status`, `increment_seconds`, `turns` — i.e. stronger players follow opening
theory longer and play distinguishable openings.

## Evaluation protocol

- **Split:** 80/20 train/test, stratified on the outcome for Task A. Fixed seed (42).
- **Selection:** 5-fold cross-validation on the **training set only** (linear vs
  gradient-boosted tree); the held-out test set is scored exactly once.
- **Baselines (mandatory):** majority-class + "higher-rated wins" (Task A);
  predict-the-mean (Task B).
- **Metrics:** accuracy / macro-F1 / log-loss + confusion matrix (Task A); MAE /
  RMSE / R² (Task B). Permutation importance for both.

See `notebooks/eda.ipynb` for distributions, target balance, feature/target
relationships, and a worked demonstration of the move-count parity leak.

## Stack & rationale

Python 3.12 · **uv** (fast, reproducible env) · pandas + pyarrow · scikit-learn
(pipelines keep preprocessing fold-safe) · FastAPI + uvicorn + pydantic v2 (typed
I/O, automatic 422s) · joblib (model persistence) · pytest · matplotlib (EDA) ·
Docker. Dependencies are kept minimal and the model is a plain scikit-learn pipeline
— clarity over cleverness.

## Run locally

```bash
# 1. Environment (installs runtime + dev tools)
uv sync --extra dev

# 2. Get the data: download the Kaggle "Chess Game Dataset (Lichess)" (datasnaek,
#    ~20k games) and place games.csv at data/raw/games.csv. (Raw data is gitignored.)

# 3. Build processed data + train both models (writes models/ + metrics.json)
uv run python -m src.data
uv run python -m src.train

# 4. Tests
uv run pytest

# 5. Serve the API
uv run uvicorn src.api.app:app --reload --port 8080
```

Example request:

```bash
curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' -d '{
  "task": "outcome",
  "features": {"rated": true, "increment_code": "10+0", "opening_eco": "C50",
               "opening_ply": 5, "white_rating": 1800, "black_rating": 1500}
}'
# -> {"task":"outcome","prediction":"white",
#     "probabilities":{"black":0.26,"draw":0.05,"white":0.69}}
```

For Task B, send `"task": "rating"` with `turns`, `winner`, `victory_status` (and the
opening/cadence fields); omit the player ratings.

## Deployment

```bash
docker build -t chess-elo .
docker run -p 8080:8080 chess-elo      # serves /health and /predict
```

The container binds to `$PORT` (default 8080), so it runs on Cloud Run (Phase 4)
with no changes. The persisted models are baked into the image.

## Limitations & next steps

- **Task A is near its ceiling for pre-game features.** Without in-game information,
  rating difference is almost all the signal; the model can't beat the higher-rated
  heuristic. Draws are effectively unpredictable here (~5% base rate).
- **Dataset skew.** Cadence is dominated by `10+0` (rapid); bullet/blitz are nearly
  absent, so `time_control` adds little.
- **No engine features.** A centipawn-loss / accuracy feature (Stockfish or `%eval`
  annotations) would likely help Task B, but is out of scope for v1 (PRD §6 stretch).
- **Modest data size** (~19k games) caps model complexity; light tuning only.

## Cost

Zero external cost: a public dataset, CPU-only scikit-learn, local training in
seconds, no paid APIs. The Docker image is a slim CPU container suitable for a
small Cloud Run instance.

## Repository structure

```
chess_elo_predictor/
├── data/processed/        # cleaned parquet (raw/ is gitignored)
├── notebooks/eda.ipynb    # EDA, runs top to bottom
├── src/
│   ├── data.py            # load + inspect schema + clean + persist
│   ├── features.py        # engineering + leakage guards + preprocessor
│   ├── train.py           # baselines, CV, models, importance, persist
│   ├── evaluate.py        # metric helpers
│   └── api/app.py         # FastAPI service
├── models/                # persisted pipelines + metrics.json
├── tests/                 # data, leakage, model-beats-baseline, API
├── Dockerfile
├── pyproject.toml         # uv-managed
├── architecture.md
└── README.md
```
