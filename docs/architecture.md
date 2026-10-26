# Architecture and System Design

## 1. Monorepo & Decoupled Execution Design
This portfolio repository utilizes a **single unified monorepo** with **independently decoupled projects**.

```
data-science-projects/
│
├── app.py                     # Central portfolio hub & multi-project router
├── requirements.txt           # Unified dependency specification
├── pyproject.toml             # Pytest & Ruff configurations
├── shared/                    # Cross-project design tokens & utility helpers
│   ├── styles.py              # Editorial CSS tokens, cards, and Plotly templates
│   └── utilities.py           # Path resolution, formatting, and file downloads
│
├── projects/
│   ├── customer_segmentation/ # Project 1: E-commerce behavioral analytics
│   ├── student_performance/   # Project 2: Educational achievement modeling
│   └── restaurant_analytics/  # Project 3: Food sales & menu engineering
│
├── docs/                      # Architectural & methodology documentation
├── scripts/                   # Local runner, test suite, and standalone exporter
└── .github/workflows/         # CI/CD automated test pipeline
```

### Architectural Principles
1. **Zero Redundant Copies:** All projects leverage common, lightweight utility helpers (`shared/`) for path discovery and clean visual styling without duplicating boilerplate.
2. **Dual Execution Modes:** Every project can be executed directly as a standalone Streamlit app or rendered dynamically inside the central portfolio router (`app.py`).
3. **Resilient Path Resolution:** `shared.utilities.resolve_data_path` resolves datasets regardless of whether a command is executed from the repository root, from inside a subproject directory, or after exporting via `scripts/publish_project.sh`.
4. **Offline Capability & Zero Cloud Overhead:** Designed to run entirely on local developer hardware (including 8 GB Apple Silicon M1 Macs) without requiring Docker, cloud relational databases, external LLMs, or paid API keys.

## 2. Shared Styling System
Adheres to strict editorial guidelines to avoid generic AI-generated aesthetics:
- **Base Background:** Neutral off-white (`#F9FAFB`).
- **Typography:** System UI typography with high contrast (`#111827`) and restrained sizes.
- **Project Color Accents:**
  - Customer Segmentation: Deep Navy / Indigo (`#1E40AF`)
  - Student Performance: Scholarly Teal / Sage (`#0F766E`)
  - Restaurant Analytics: Warm Terracotta / Sienna (`#C2410C`)
- **Plotly Integration:** Chart templates inherit transparent backgrounds, light `#F3F4F6` gridlines, readable font families, and uncluttered legends.

## 3. Data Flow & State Management
- **In-Memory Caching:** Heavy I/O and model training procedures utilize `@st.cache_data` decorators with sensible cache invalidation keys.
- **Stateless Analysis Layer:** Functional pipelines in `analysis.py` and `modeling.py` receive pandas DataFrames and return standard Python dictionaries, Series, or DataFrames, maintaining clean separation between computation and UI presentation.
