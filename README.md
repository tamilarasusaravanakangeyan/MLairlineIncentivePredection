# ML Airline Incentive Prediction

[![API CI](https://github.com/tamilarasusaravanakangeyan/MLairlineIncentivePredection/actions/workflows/api-ci.yml/badge.svg)](https://github.com/tamilarasusaravanakangeyan/MLairlineIncentivePredection/actions/workflows/api-ci.yml)

End-to-end machine learning implementation for airline travel-agency incentives using PostgreSQL historical data.

## Changelog

- 2026-02-26: Clarified Option A/B/C evaluation metrics and their operational interpretation.

## What is implemented

### Option A — Classification
Predicts the next quarter incentive tier for each agency.

- Target: `next_quarter_tier_id`
- Model: `XGBClassifier` (multiclass)
- Metrics:
  - F1 (weighted): overall class assignment quality under imbalance
  - ROC-AUC (OvR, weighted): class separability across decision thresholds
  - Classification report: per-tier precision/recall/F1 to detect weak tiers

### Option B — Regression
Predicts the next quarter revenue target for each agency.

- Target: `next_quarter_revenue_target`
- Model: `XGBRegressor`
- Metrics:
  - RMSE: sensitivity to large forecast misses
  - MAE: average absolute miss in business revenue units
  - R2: variance explained versus baseline behavior

### Option C — Next Best Incentive Recommendation
Recommends:

- Incentive program
- Incentive tier
- Commission percentage

using a combined score from predicted tier confidence + predicted revenue target + commission attractiveness.

- Evaluation in practice:
  - Tracks business metrics such as recommendation acceptance rate, revenue uplift, and commission efficiency
  - Uses Option A + B model quality as upstream reliability signals for recommendation trust

## Data sources

Main PostgreSQL tables:

- `airlines`
- `routes`
- `transactions`
- `travel_agencies`
- `agency_performance`
- `incentive_programs`
- `incentive_tiers`
- `incentive_redemptions`

Default DB URL:

`postgresql://postgres:postgres@localhost:5432/airline_incentives`

## Feature engineering highlights

- Rolling windows: 3-month and 6-month
- Lag features: month-level lags
- Growth features: revenue and ticket growth
- RFM-style features: recency, frequency, monetary
- Diversity features: airline and route diversity
- Seasonality features: month and quarter
- Leakage-safe design: only historical data is used for feature construction

## Notebook implementation

Primary notebook:

- `notebooks/next_best_incentive_tier_model.ipynb`

Notebook sections include:

1. Imports
2. Config
3. DB connection
4. Data extraction
5. Feature engineering
6. Label generation
7. Time-based split (last 3 months validation)
8. Model training
9. Evaluation
10. Feature importance
11. SHAP explainability
12. Artifact save
13. Option B regression training/evaluation/save
14. Option C recommendation logic

## Generated artifacts

Produced by notebook training in `notebooks/artifacts/`:

- `next_tier_xgb.joblib`
- `feature_columns.joblib`
- `label_mapping.joblib`
- `next_revenue_xgb_regressor.joblib`

## API implementation

Main API module:

- `api/main.py`

Endpoints:

- `GET /health`
- `GET /predict/{agency_id}?as_of_date=YYYY-MM-DD` (Option A)
- `GET /predict-revenue/{agency_id}?as_of_date=YYYY-MM-DD` (Option B)
- `GET /recommend/{agency_id}?as_of_date=YYYY-MM-DD` (Option C)

Detailed API docs:

- `api/README.md`

## Streamlit dashboard

Dashboard module:

- `ui/streamlit_dashboard/app.py`

Runs on a separate port and visualizes Option A/B/C outputs in tables and charts.

Start dashboard:

```bash
pip install -r ui/streamlit_dashboard/requirements.txt
streamlit run ui/streamlit_dashboard/app.py --server.port 8501
```

Or use helper scripts from repository root:

- Windows: `./run_dashboard.ps1`
- Linux/macOS: `chmod +x ./run_dashboard.sh && ./run_dashboard.sh`

## PostgreSQL explorer dashboard (new)

Database + model comparison dashboard module:

- `ui/db_compare_dashboard/app.py`

This dashboard connects directly to PostgreSQL to browse all tables and also compares model outputs (Option A/B/C API) against historical transaction aggregates.

Start database explorer dashboard:

```bash
pip install -r ui/db_compare_dashboard/requirements.txt
streamlit run ui/db_compare_dashboard/app.py --server.port 8502
```

Or use helper scripts from repository root:

- Windows: `./run_db_dashboard.ps1`
- Linux/macOS: `chmod +x ./run_db_dashboard.sh && ./run_db_dashboard.sh`

## Quick start

### Install dependencies

```bash
pip install -r api/requirements.txt
```

### Run API

```bash
uvicorn api.main:app --reload
```

### One-command scripts

- Windows: `./run_api.ps1`
- Linux/macOS: `chmod +x ./run_api.sh && ./run_api.sh`

## Validation and CI

Local smoke test:

```bash
python api/smoke_test.py
```

CI workflow:

- `.github/workflows/api-ci.yml`

The workflow runs syntax checks and smoke tests for API changes.

## Repository structure

```text
.github/
  instructions/
  prompts/
  workflows/
api/
  main.py
  README.md
  requirements.txt
  smoke_test.py
notebooks/
  next_best_incentive_tier_model.ipynb
run_api.ps1
run_api.sh
run_dashboard.ps1
run_dashboard.sh
run_db_dashboard.ps1
run_db_dashboard.sh
ui/
  db_compare_dashboard/
    app.py
    README.md
    requirements.txt
  streamlit_dashboard/
    app.py
    README.md
    requirements.txt
README.md
```

## Notes

- Time-based splitting is used; random split is not used.
- Feature extraction and targets are designed to avoid future leakage.
- If regression artifacts are missing, Option B and Option C API endpoints return `503` until regression training is run.
