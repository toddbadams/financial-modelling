# Financial Modelling

A Streamlit-based educational app for learning quantitative finance concepts. It uses interactive visualizations, live market data, and structured educational content to teach topics like asset risk, return analysis, normalization, and portfolio optimization.

## Repository Structure

- **`/src`** — Application source code (Streamlit UI, data layer, reusable components)
- **`/workbooks`** — Resource material used as background information when creating the application (Jupyter notebooks, ML classifier module, educational notes)
- **`/ai_context`** — Context markdown files for AI agents

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
streamlit run src/ui/app.py
```

Streamlit will start a local server, and the console prints a URL (usually `http://localhost:8501`) that you can open in your browser. Control-C stops the app when you're done.

## 4. Quantitative Finance Learning Websites

For those looking to enhance their quantitative finance skills, there are several reputable websites and platforms that offer a range of resources and courses. Here are some notable options:

[Quants Hub](https://quantshub.com/): A comprehensive online resource for Quantitative Analysts, Risk Managers, Data Scientists, and more, offering both online training and a rich library of content for self-paced learning. 

[QuantStart](https://www.quantstart.com/articles/Free-Quantitative-Finance-Resources/): A big list of free quantitative finance resources, including ebooks, slides, courses, videos, and data, all available for free or with a free signup. 

[QuantEdX](https://quantedx.com/): An open-access platform for quantitative finance education, providing expert-led online courses and interactive notebooks. 

[Best Quantitative Finance courses (2026) ranked by Bankers](https://www.bankersbyday.com/quantitative-finance-courses-certifications/): A detailed list of courses covering various aspects of quantitative finance, including traditional and behavioral finance theory, technical analysis, and algorithmic trading. 

[Quantitative Finance Stack Exchange](https://quant.stackexchange.com/): A community forum where users can ask questions and share knowledge on quantitative finance topics. 

[Best Quantitative Finance Courses & Certificates | Coursera](https://www.coursera.org/courses?query=quantitative%20finance&msockid=142abda7c13a66dc1b3fa86bc0da671a): A platform offering a variety of courses on financial modeling, risk assessment, portfolio optimization, and algorithmic trading strategies. 
