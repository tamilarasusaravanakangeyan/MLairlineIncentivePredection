# PostgreSQL Data Explorer + Output Comparison Dashboard

This Streamlit dashboard is a separate UI to:

- Browse all PostgreSQL tables from `airline_incentives`
- View data in tabular format and chart selected columns
- Compare model outputs (Option A/B/C API) with historical DB aggregates

## Run on separate port

From repository root:

```bash
pip install -r ui/db_compare_dashboard/requirements.txt
streamlit run ui/db_compare_dashboard/app.py --server.port 8502
```

Open in browser:

- `http://127.0.0.1:8502`

## Inputs

- PostgreSQL URL (default):
  - `postgresql://postgres:postgres@host.docker.internal:5432/airline_incentives`
- API base URL (default):
  - `http://127.0.0.1:8000`
- As-of date and agency IDs for comparison

## Notes

- Keep FastAPI running to compare outputs in the `Compare Output` tab.
- The `Database` tab works directly from PostgreSQL and does not require API availability.
