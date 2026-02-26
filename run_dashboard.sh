#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r ui/streamlit_dashboard/requirements.txt
python -m streamlit run ui/streamlit_dashboard/app.py --server.port 8501
