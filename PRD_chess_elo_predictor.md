# PRD — Chess Game-Outcome & Elo Predictor

> Hand this file to Claude Code as the spec for the build. Work through the **Build order**
> at the end. Ask me before making irreversible changes or installing anything heavyweight.

## 1. Goal

A small, honest, well-engineered ML project that predicts, from features of a completed
Lichess game:

- **Task A — Outcome** (classification): white win / black win / draw.
- **Task B — Rating** (regression): the average player rating of the game.

It must look like production engineering, not a notebook dump: reproducible pipeline, baselines,
real metrics on a held-out test set, a typed FastAPI service, Docker packaging, tests, and clean
docs. This is a fundamentals project (Phase 1 of my AI-Engineer roadmap) — favor clarity and
correctness over cleverness.

## 2. Quality bar (non-negotiable)

- **Never fabricate numbers.** Every metric in the README comes from an actual eval run. If
  something isn't measured, write a clearly-marked `TODO`, not a guess.
- **No data leakage.** See §5. Guard against it explicitly and add a test that enforces it.
- **Reproducible.** Fixed random seed; one command trains and evaluates end to end.
- **Beat a baseline.** A model that doesn't beat a trivial baseline is reported as such, honestly.

## 3. Scope

**In scope (v1):** data ingest + cleaning, EDA notebook, feature engineering, both models with
baseline comparison and cross-validation, model persistence, FastAPI inference service, Dockerfile,
tests, README + architecture doc.

**Out of scope (v1):** cloud deployment (Phase 4), any LLM/GenAI, a frontend, the "accuracy"
feature (stretch — see §6), hyperparameter-tuning heroics (a light grid/`HalvingRandomSearchCV`
is plenty).

## 4. Data

- Use a public Lichess game dataset. Default to the Kaggle "Chess Game Dataset (Lichess)"
  (~20k games); the Lichess open database (`database.lichess.org`) or API are acceptable
  alternatives.
- **Inspect the actual schema before coding features** — do not hardcode column names from
  assumptions. Print the columns, dtypes, missingness, and a sample, and adapt.
- Persist a cleaned/processed dataset to `data/processed/` so training is reproducible. Keep raw
  data out of git (`.gitignore`), but document exactly how to obtain it in the README.

## 5. ML framing & leakage guards (read carefully)

Framing matters more than model choice here.

**Task A — Outcome.** Frame as: *given the conditions of a completed game (both players' ratings,
opening, time control, game length), predict who won.* This is descriptive, not a live in-game
predictor — state that plainly in the README.
- **Must exclude** any field that encodes the result: `winner`, `victory_status`
  (mate/resign/draw/outoftime), and anything derived from them. Including these is leakage.
- `rating_diff` and `avg_rating` are legitimate (known before the game).
- `move_count` is known only after the game; it's acceptable under the descriptive framing, but
  note it in the README as a post-hoc feature.
- Add a unit test asserting the leakage columns are absent from the training feature matrix.

**Task B — Rating.** Predict `avg_rating` from opening, time control, move count, outcome, and
victory_status. All of these are fair inputs here.

## 6. Features

Engineer from whatever the schema provides; likely:
- `rated` (bool)
- Time control → parse base seconds + increment → bucket into bullet / blitz / rapid / classical.
- Opening (ECO code and/or name) → encode (one-hot top-N openings + an "other" bucket, or target
  encoding done **inside** cross-validation folds to avoid leakage).
- `move_count` / number of turns.
- `white_rating`, `black_rating` → derive `rating_diff`, `avg_rating` (avg is Task B's target, so
  don't use it as a feature for Task B).

**Stretch (do NOT block v1):** an `accuracy` / centipawn-loss feature. This requires engine
analysis (Stockfish) or games that already carry `%eval` annotations — flag it as future work
unless eval data is readily available.

## 7. Modeling & evaluation

- **Split:** train/test (e.g. 80/20), stratified on the outcome for Task A. Use cross-validation
  on the training set for model selection.
- **Baselines (mandatory):**
  - Task A: majority-class predictor, and a "higher-rated player wins" heuristic.
  - Task B: predict the mean rating.
- **Models:** start with a linear model (LogisticRegression / Ridge), then a tree ensemble
  (`HistGradientBoosting` or RandomForest). Compare; pick the best on CV; report on the held-out
  test set only once.
- **Metrics:**
  - Task A: accuracy, macro-F1, confusion matrix, log-loss.
  - Task B: MAE, RMSE, R².
- Produce **feature importances** (or permutation importance) and a short written takeaway.
- Persist the chosen model(s) with `joblib` to `models/`, and write metrics to
  `models/metrics.json` so the README/tests can read them rather than restating by hand.

## 8. API (FastAPI)

- Endpoints: `GET /health` → `{"status":"ok"}`; `POST /predict`.
- `/predict` takes a typed Pydantic request of raw game features and a `task` field
  (`"outcome"` or `"rating"`), and returns:
  - outcome → predicted class + class probabilities;
  - rating → predicted value.
- Load the persisted model **once at startup**, not per request.
- Validate inputs; return a clean 422 on bad input. Include an example request in the README.

## 9. Packaging (Docker)

- A slim Python image that binds to `$PORT` (default 8080) so it's Cloud-Run-ready for Phase 4
  with zero changes. `docker build` then `docker run -p 8080:8080` must serve `/predict`.

## 10. Tests (pytest)

- Data/feature tests: cleaning behaves; **leakage guard** test (excluded columns absent).
- Model test: a tiny fixture trains and the model beats the majority/mean baseline on it.
- API tests: `/health` returns ok; `/predict` happy path returns the right shape; bad input → 422.

## 11. Repository structure

```
chess-elo-predictor/
├── data/                  # raw/ (gitignored) + processed/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data.py            # load + clean + inspect schema
│   ├── features.py        # feature engineering (+ leakage guards)
│   ├── train.py           # train + CV + evaluate vs baselines + persist
│   ├── evaluate.py        # metrics helpers
│   └── api/
│       └── app.py         # FastAPI service
├── models/                # persisted model + metrics.json
├── tests/
├── Dockerfile
├── pyproject.toml         # managed with uv
├── README.md
└── architecture.md
```

## 12. Tech stack

Python 3.12 · uv · pandas · scikit-learn · FastAPI · uvicorn · pydantic v2 · joblib · pytest ·
matplotlib (EDA only) · Docker. Keep dependencies minimal.

## 13. Definition of Done

- [ ] `uv run python -m src.train` trains both tasks, prints metrics vs baselines, and writes
      `models/metrics.json` — reproducibly (fixed seed).
- [ ] Held-out test metrics reported for both tasks; both beat their baselines (or it's stated honestly).
- [ ] Leakage-guard test passes.
- [ ] `docker build` + `docker run` serves `/health` and `/predict`.
- [ ] `pytest` is green.
- [ ] EDA notebook runs top to bottom.
- [ ] README.md + architecture.md complete, metrics pulled from the real run, remaining gaps marked `TODO`.
- [ ] Everything committed to a git repo.

## 14. Build order (work through these)

1. Scaffold repo, `pyproject.toml` (uv), structure, `.gitignore`, `git init`.
2. `src/data.py`: obtain + load + **inspect schema** + clean; save processed data.
3. `notebooks/eda.ipynb`: distributions, target balance, feature/target relationships.
4. `src/features.py`: engineering + **leakage guards** for Task A.
5. `src/train.py` + `src/evaluate.py`: baselines, CV, both models, metrics, feature importance, persist.
6. `src/api/app.py`: FastAPI loading the persisted model; typed schemas.
7. Dockerfile; verify the container serves `/predict`.
8. `tests/`: data, leakage, model, API.
9. README.md + architecture.md — **use my installed `portfolio-project-packager` skill** to
   generate these; if it's not available, follow the section list it defines (overview,
   architecture, key design decisions, results & metrics, evaluation, stack & rationale, run
   locally, deployment, limitations, cost, repo structure). Pull all numbers from `models/metrics.json`.
10. Final pass against the Definition of Done; list any remaining `TODO`s.

## 15. Notes for the implementer

- Prefer clear, typed, well-named code over abstraction. This is read by interviewers.
- Commit in logical chunks with meaningful messages — the git history is part of the story.
- If a modeling result is mediocre, that's fine — report it honestly and note what you'd try next.
  An honest, well-engineered B+ project beats a suspiciously perfect one.
