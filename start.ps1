# Start the RAG API backend
# Usage: .\start.ps1

Write-Host "Starting RAG API backend..." -ForegroundColor Cyan

# Activate virtual environment if it exists
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
    Write-Host "Virtual environment activated." -ForegroundColor Green
}

# Start uvicorn with correct host binding
uvicorn main:app --reload --host 0.0.0.0 --port 8000
