# Data Science Projects Portfolio

[![Tests & Code Quality](https://github.com/deeppatel632/data-science-projects-portfolio/actions/workflows/tests.yml/badge.svg)](https://github.com/deeppatel632/data-science-projects-portfolio/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.24+-purple.svg)](https://plotly.com/)

A portfolio containing three intermediate-level data science and applied analytics projects built with clean Python engineering, interactive Streamlit interfaces, publication-grade Plotly visualizations, rigorous leakage prevention, and complete test suites.

---

## Portfolio Projects Overview

| Project | Domain | Dataset Source | Primary Methods | Key Deliverable |
| :--- | :--- | :--- | :--- | :--- |
| **1. Customer Behavioral Segmentation & Purchase Intent** | E-Commerce Analytics | UCI Machine Learning Repository (12,330 sessions) | K-Means ($k=4$), PCA 2D Projection, Logistic Regression, Random Forest | Unsupervised behavioral profiling & leakage-free conversion prediction with threshold simulator |
| **2. Student Academic Performance Analysis** | Educational Data Science | UCI Machine Learning Repository (1,044 students across Math & Portuguese) | Exploratory Correlation, Dummy Baseline, Ridge Linear Regression, Random Forest | Early-warning vs. mid-term trajectory scenario regression with non-causal ethical governance |
| **3. Restaurant Performance & Food Sales Analysis** | Commercial Business Intelligence | Calibrated Synthetic POS Dataset (14,785 line items, 7,500 orders) | Financial KPI Aggregation, Menu Engineering Matrix, ABC Pareto Analysis, Rolling Trend Forecasting | Multi-outlet profitability dashboard, sales channel mix, rush hour analysis & custom CSV uploader |

---

## Repository Directory Structure

```
data-science-projects/
│
├── app.py                     # Central portfolio application & interactive router
├── requirements.txt           # Verified Python dependencies
├── pyproject.toml             # Pytest & Ruff linter configuration
├── README.md                  # Comprehensive portfolio documentation
├── LICENSE                    # MIT open-source license
├── .gitignore                 # Python & Streamlit environment exclusions
│
├── .streamlit/
│   └── config.toml            # Editorial light design theme tokens
│
├── shared/                    # Cross-project styling and utility components
│   ├── __init__.py
│   ├── styles.py              # Editorial CSS tokens, typography, and Plotly templates
│   └── utilities.py           # Path resolution, currency/number formatters, downloads
│
├── projects/
│   ├── customer_segmentation/ # Project 1: E-commerce behavioral analytics
│   │   ├── app.py             # Standalone & embedded Streamlit dashboard
│   │   ├── data.py            # UCI data loader & reproducible synthetic fallback
│   │   ├── analysis.py        # EDA, session conversions, monthly seasonality
│   │   ├── modeling.py        # K-Means, PCA, Logistic Regression, Random Forest
│   │   ├── visualizations.py  # Plotly publication charts
│   │   ├── README.md          # Project-specific technical documentation
│   │   ├── data/              # Bundled UCI dataset
│   │   └── tests/             # Pytest test suite
│   │
│   ├── student_performance/   # Project 2: Educational achievement modeling
│   │   ├── app.py             # Standalone & embedded Streamlit dashboard
│   │   ├── data.py            # UCI Math & Portuguese data loader & fallback
│   │   ├── analysis.py        # Academic overview, study habit correlations
│   │   ├── modeling.py        # Early-warning vs mid-term regression pipelines
│   │   ├── visualizations.py  # Grade distributions, trajectories, actual vs predicted
│   │   ├── README.md          # Project-specific technical documentation
│   │   ├── data/              # Bundled UCI datasets
│   │   └── tests/             # Pytest test suite
│   │
│   └── restaurant_analytics/  # Project 3: Food sales & menu engineering
│       ├── app.py             # Standalone & embedded Streamlit dashboard
│       ├── data.py            # Deterministic POS generator & schema validator
│       ├── analysis.py        # Financial KPIs, ABC analysis, rolling forecast
│       ├── visualizations.py  # Sales trends, outlet comparisons, menu scatter
│       ├── README.md          # Project-specific technical documentation
│       ├── data/              # Bundled sample transaction dataset
│       └── tests/             # Pytest test suite
│
├── docs/                      # Architectural, dataset, and methodology documentation
│   ├── architecture.md        # System design & monorepo decoupling
│   ├── datasets.md            # Provenance, citations, and schemas
│   ├── modeling_notes.md      # ML pipelines, target leakage prevention, scenarios
│   ├── limitations.md         # Non-causal caveats and ethical guidelines
│   └── screenshots/           # Application verification captures
│
├── scripts/                   # Automations & developer utilities
│   ├── run.sh                 # Local launch script (macOS/Linux)
│   ├── run_tests.sh           # Test suite & Ruff lint execution
│   └── publish_project.sh     # Standalone project export and packaging utility
│
└── .github/
    └── workflows/
        └── tests.yml          # GitHub Actions CI workflow
```

---

## Prerequisites & Local Setup

This portfolio runs entirely on local consumer hardware (including Apple Silicon M1 Macs with 8 GB RAM, Linux, or Windows). No Docker containers, paid cloud databases, or GPU accelerators are required.

### 1. Clone the Repository
```bash
git clone https://github.com/deeppatel632/data-science-projects-portfolio.git
cd data-science-projects-portfolio
```

### 2. Create and Activate Virtual Environment

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Applications

### Central Portfolio Application
Launch the central interactive hub to explore all three projects from a single server:
```bash
streamlit run app.py
```
*Or use the launch script on macOS/Linux:*
```bash
./scripts/run.sh
```

### Launching Projects Individually
Each project contains its own self-contained entry point and can be launched directly:

```bash
# Project 1: Customer Behavioral Segmentation
streamlit run projects/customer_segmentation/app.py

# Project 2: Student Academic Performance
streamlit run projects/student_performance/app.py

# Project 3: Restaurant Sales & Performance
streamlit run projects/restaurant_analytics/app.py
```

---

## Running Tests & Code Quality

Run the automated test suite covering all 3 projects:
```bash
pytest -v
```
Run linting checks:
```bash
ruff check .
```
Or execute the automated test runner:
```bash
./scripts/run_tests.sh
```

---

## Standalone Project Handover & Export

To extract an individual project into its own independent standalone repository for external handover:
```bash
./scripts/publish_project.sh <project-name>
```

**Supported Project Names:**
- `customer-segmentation`
- `student-performance`
- `restaurant-analytics`

**Example:**
```bash
# Exports to dist/customer-segmentation and runs standalone test suite
./scripts/publish_project.sh customer-segmentation

# Exports to a custom directory and initializes git
./scripts/publish_project.sh student-performance --output-dir /path/to/standalone-repo --init-git
```
The export script bundles the necessary `shared/` modules, config files, requirements, documentation, and dataset while excluding `.venv` and cache artifacts.

---

## Dataset Attributions & Licensing

- **UCI Online Shoppers Purchasing Intention Dataset:**
  - Citation: Sakar, C.O., Polat, S.O., Katircioglu, M. et al. (2019). *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.* Neural Computing and Applications, 31, 6893–6908.
  - License: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **UCI Student Performance Dataset:**
  - Citation: Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance.* In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal.
  - License: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Restaurant Demonstration Dataset:**
  - Deterministically simulated operational data generated with seed 42.

---

## License

This project is licensed under the [MIT License](LICENSE).
