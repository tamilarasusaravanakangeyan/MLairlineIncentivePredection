$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

& $python -m pip install -r ui/db_compare_dashboard/requirements.txt
& $python -m streamlit run ui/db_compare_dashboard/app.py --server.port 8502
