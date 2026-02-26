$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

& $python -m pip install -r ui/streamlit_dashboard/requirements.txt
& $python -m streamlit run ui/streamlit_dashboard/app.py --server.port 8501
