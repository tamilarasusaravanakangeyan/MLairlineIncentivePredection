#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_EXE=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  PYTHON_EXE=".venv/Scripts/python.exe"
else
  echo "Python virtual environment not found at .venv. Create it first, then rerun this script." >&2
  exit 1
fi

"$PYTHON_EXE" -m pip install -r api/requirements.txt
"$PYTHON_EXE" -m uvicorn api.main:app --reload
