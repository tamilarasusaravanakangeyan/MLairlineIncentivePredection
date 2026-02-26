# Next Best Incentive Tier API

[![API CI](https://github.com/tamilarasusaravanakangeyan/MLairlineIncentivePredection/actions/workflows/api-ci.yml/badge.svg)](https://github.com/tamilarasusaravanakangeyan/MLairlineIncentivePredection/actions/workflows/api-ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com/)

FastAPI service for:

- Option A: next-quarter tier classification
- Option B: next-quarter revenue target regression
- Option C: next-best incentive recommendation

## Changelog

- 2026-02-26: Clarified Option A/B/C evaluation metrics and operational interpretation in project documentation.

## Endpoints

- `GET /health`
- `GET /predict/{agency_id}?as_of_date=YYYY-MM-DD`
- `GET /predict-revenue/{agency_id}?as_of_date=YYYY-MM-DD`
- `GET /recommend/{agency_id}?as_of_date=YYYY-MM-DD`

## Response Contract

`/predict` returns:

```json
{
  "agency_id": 1,
  "as_of_date": "2026-02-01",
  "predicted_tier_id": 2,
  "confidence_score": 0.3764871656894684
}
```

`/predict-revenue` returns:

```json
{
  "agency_id": 1,
  "as_of_date": "2026-02-01",
  "predicted_revenue_target": 250000.0
}
```

`/recommend` returns:

```json
{
  "agency_id": 1,
  "as_of_date": "2026-02-01",
  "program_id": 2,
  "program_name": "Growth Incentive Program",
  "tier_id": 2,
  "tier_name": "Gold",
  "tier_level": 2,
  "commission_percentage": 4.5,
  "predicted_tier_id": 2,
  "confidence_score": 0.82,
  "predicted_revenue_target": 250000.0
}
```

## Prerequisites

- Python 3.11+
- Trained artifacts from notebook run in `notebooks/artifacts/`:
  - `next_tier_xgb.joblib`
  - `feature_columns.joblib`
  - `label_mapping.joblib`
  - `next_revenue_xgb_regressor.joblib` (required for `/predict-revenue` and `/recommend`)

## Install

```bash
pip install -r api/requirements.txt
```

## Run

```bash
uvicorn api.main:app --reload
```

## Windows One-Command Run

From repository root:

```powershell
.\run_api.ps1
```

This script installs API dependencies and starts Uvicorn using the workspace virtual environment.

## Linux/macOS One-Command Run

From repository root:

```bash
chmod +x ./run_api.sh
./run_api.sh
```

This script installs API dependencies and starts Uvicorn using the workspace virtual environment.

## Environment Variables

- `DATABASE_URL` (optional)
  - Default: `postgresql://postgres:postgres@host.docker.internal:5432/airline_incentives`
- `ARTIFACT_DIR` (optional)
  - Default: `notebooks/artifacts`

## Example Requests

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Predict

```bash
curl "http://127.0.0.1:8000/predict/1?as_of_date=2026-02-01"
```

### Predict Revenue Target

```bash
curl "http://127.0.0.1:8000/predict-revenue/1?as_of_date=2026-02-01"
```

### Recommend Incentive

```bash
curl "http://127.0.0.1:8000/recommend/1?as_of_date=2026-02-01"
```

## Notes

- Feature extraction is leakage-safe and uses only historical data up to `as_of_date`.
- If no agency history exists up to the provided date, the API returns `404`.
- If regression artifact is missing, `/predict-revenue` and `/recommend` return `503` until the regression model is trained and saved.
