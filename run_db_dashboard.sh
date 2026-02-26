#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r ui/db_compare_dashboard/requirements.txt
python -m streamlit run ui/db_compare_dashboard/app.py --server.port 8502
