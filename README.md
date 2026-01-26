

# Project setup

## 1. Install and configure the project dependencies

1. Make sure [Poetry](https://python-poetry.org/docs/) is installed and available in your shell.
2. Set a local cache directory (prevents repeated downloads) and install:

   ```powershell
   $projectCache = Join-Path (Get-Location) '.cache\poetry'
   if (-not (Test-Path $projectCache)) { New-Item -ItemType Directory -Path $projectCache | Out-Null }
   $Env:POETRY_CACHE_DIR = $projectCache
   poetry install -v
   ```

   This will create the `.venv` alongside the project and install `streamlit` plus any other dependencies declared in `pyproject.toml`.

## 2. Launch the virtual environment

Use one of the following once the install finishes:

- `poetry shell`
- `.\.venv\Scripts\Activate` (PowerShell) / `.\.venv\Scripts\activate.bat` (Command Prompt)

After activation, verify with `python --version` or `pip list` to confirm you're inside the `.venv`.

## 3. Run the Streamlit app

From the project root (still within the virtual environment) run:

```powershell
streamlit run week1/x.py
```

Streamlit will start a local server, and the console prints a URL (usually `http://localhost:8501`) that you can open in your browser. Control-C stops the app when you're done.
