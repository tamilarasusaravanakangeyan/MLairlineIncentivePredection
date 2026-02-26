$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $pythonExe)) {
    throw "Python virtual environment not found at .venv. Create it first, then rerun this script."
}

& $pythonExe -m pip install -r 'api/requirements.txt'
& $pythonExe -m uvicorn api.main:app --reload
