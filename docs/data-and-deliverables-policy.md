# Data and Deliverables Policy

## Data handling

When a course project uses local data files, organize them as:

```
data/
├── raw/       # original source data — treat as immutable
└── output/    # cleaned, feature-ready, or modeling-ready data
```

Keep raw data unmodified. Output data should be reproducible from upstream steps.

## Deliverables

Each course project should produce:

- Project `README.md` — objective, data description, how to run
- Notebooks — marimo (preferred) or Jupyter, with clear narrative
- HTML report — exported from the notebook (`just export-html <path>`)
- Reproducible environment — `pyproject.toml` + `uv.lock` + `.python-version`

## Git policy

Commit:
- source code, notebooks, reports
- project metadata and lockfiles

Do not commit:
- `.venv/`, caches, notebook checkpoints
- large derived artifacts unless required
- secrets or credentials
