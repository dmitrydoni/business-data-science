# Business Data Science

Monorepo for academic data science and ML courses. Each course under `courses/` is an independent `uv` project with its own environment, dependencies, and notebooks.

## Courses

| Course | Status |
|--------|--------|
| [Math Statistics](courses/math-statistics/) | Completed |
| [Business Statistics](courses/business-statistics/) | Upcoming |

## Quick start

```
just help          # show available commands
just new <name>    # scaffold a new course project
just tree          # show repo structure
```

## Repo layout

```
business-data-science/
├── courses/           # independent uv projects, one per course
├── docs/              # shared documentation and policies
├── templates/         # course scaffolding template
├── utils/             # reusable Python utilities (EDA, modeling)
└── justfile           # CLI for common operations
```

## Notebook format

New notebooks use [marimo](https://marimo.io/). Legacy Jupyter notebooks can be converted:

```
just convert courses/<course>/hands_on/notebook.ipynb
```

## Documentation

- [Repository architecture](docs/repo-architecture.md)
- [uv project policy](docs/uv-project-policy.md)
- [Data and deliverables policy](docs/data-and-deliverables-policy.md)
