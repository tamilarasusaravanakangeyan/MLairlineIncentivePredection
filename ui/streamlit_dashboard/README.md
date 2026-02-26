# Streamlit Dashboard

This dashboard runs on a separate port from the API and visualizes model outcomes for:

- Option A: tier classification
- Option B: revenue target regression
- Option C: recommendation (program + tier + commission)

## Run

From repository root:

```bash
pip install -r ui/streamlit_dashboard/requirements.txt
streamlit run ui/streamlit_dashboard/app.py --server.port 8501
```

Open in browser:

- `http://127.0.0.1:8501`

## Requirement

API should be running (default: `http://127.0.0.1:8000`).

You can change API URL from the sidebar in the dashboard.
