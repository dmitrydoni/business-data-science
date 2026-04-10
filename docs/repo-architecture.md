# Repository Architecture

## Structure

```
business-data-science/
├── CLAUDE.md
├── README.md
├── .gitignore
├── justfile
├── docs/
│   ├── repo-architecture.md
│   └── data-and-deliverables-policy.md
├── templates/
│   └── course-template/
│       ├── README.md
│       ├── pyproject.toml
│       ├── hands_on/
│       ├── case_study/
│       └── project/
├── utils/
│   ├── eda_utils.py
│   └── model_utils.py
└── courses/
    ├── math-statistics/
    └── business-statistics/
```

## Design principles

### One repo, many independent projects

Each course under `courses/` is an independent `uv` project with its own:

- `pyproject.toml`
- `uv.lock`
- `.python-version`
- `.venv` (local, not committed)

Do not use a `uv` workspace. Projects may differ in Python version, libraries, and tooling.

### Shared assets at the top level

- `docs/` — policies and architecture documentation
- `templates/` — course scaffolding
- `utils/` — reusable Python modules (EDA, modeling helpers)

### Course project layout

Each course follows a standard internal structure:

```
courses/<course-name>/
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── hands_on/        # lesson exercises and practice notebooks
├── case_study/      # instructor-led case studies
└── project/         # capstone or graded project
```

### Notebook format

New notebooks use [marimo](https://marimo.io/). Legacy Jupyter notebooks are preserved as-is but can be converted with `just convert <path>`.

## Creating a new course

```
just new <course-name>
cd courses/<course-name>
uv python pin <version>
uv add <dependencies>
uv lock
```

## What to commit

Per project:
- `pyproject.toml`, `uv.lock`, `.python-version`
- Source code, notebooks, reports

Do not commit:
- `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`
- Large derived artifacts, secrets, credentials
