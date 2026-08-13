# CV Agent — Backend

FastAPI + SQLModel + PostgreSQL backend for the production computer vision agent project.

## What's here

```
app/
  main.py            FastAPI app, CORS, global error handlers
  config.py           Settings (env vars, see .env.example)
  database.py          Postgres engine/session (SQLModel)
  models.py            `predictions` table
  schemas.py            Request/response Pydantic models
  api/
    health.py           GET /health
    predictions.py        POST /api/v1/predict
    history.py             GET /api/v1/predictions, GET /api/v1/predictions/{id}
    stats.py                 GET /api/v1/stats
    model_info.py             GET /api/v1/model
  services/
    inference.py       Real ConvNeXt-Tiny inference, auto-falls back to mock
    prediction_service.py   DB read/write logic
tests/                 6 passing tests (health, predict, validation, history, stats, DB insert)
```

## Run locally

```bash
uv sync
cp .env.example .env   # edit DATABASE_URL to point at your Postgres

uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

## Run tests

```bash
uv run pytest -q
```

## Real model integration status

`app/services/inference.py` now implements the **real** ConvNeXt-Tiny stone
classifier (matching `training/train.py` and `training/evaluate.py` exactly:
same architecture, same 224x224 + ImageNet-normalization preprocessing, same
`state_dict`-only loading). It was verified end-to-end with a synthetic
checkpoint of the identical shape (state_dict loads cleanly, forward pass
produces valid softmax probabilities, full `/predict` -> DB -> `/predictions`
flow works over HTTP).

**One manual step left:** drop the real trained files in place —

```
stone-classification/
└── models/
    ├── model.pt       <- the file your CV teammate sent you (~106MB)
    └── labels.json     <- already included in this delivery, real class names
```

Then locally, set in `backend/.env`:
```
MODEL_PATH=../models/model.pt
LABELS_PATH=../models/labels.json
```

If `models/model.pt` is missing, the service automatically falls back to mock
mode (random-but-valid scores) — nothing breaks, it just won't be using the
real model yet. No code changes needed either way; only the two files.

Also worth getting from the CV teammate: `reports/model_metrics.json`
(produced by running `training/evaluate.py`) — not required for the API to
run, but needed for the README/rubric metrics section.

## Docker

```bash
docker build -t cv-agent-backend .
docker run --env-file .env -p 8000:8000 cv-agent-backend
```

Inside Docker Compose, use the service name for Postgres, not `localhost`:

```
DATABASE_URL=postgresql+psycopg://cvuser:cvpassword@postgres:5432/cvapp
```
