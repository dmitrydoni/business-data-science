# Machine Learning Basics

## Overview

Introduction to supervised classification — three core algorithms (K-Nearest Neighbors, Naive Bayes, Support Vector Machines), classification evaluation, and end-to-end project work. Walsh DBA program, supervised learning weeks 5–6.

## Structure

```
.
├── pyproject.toml
├── uv.lock
├── .python-version
├── theory/         # lecture decks and supplementary material (PDFs)
├── hands_on/       # guided lesson notebooks (KNN, Naive Bayes, SVM)
├── case_study/     # instructor-led case studies (MLS-1 Machine Failure, MLS-2 WHO)
├── tests/          # weekly quiz screenshots
└── project/        # capstone: INN Hotels booking cancellation prediction
```

## Environment

```
uv sync
```

Python 3.14+. Dependencies: numpy, pandas, scipy, matplotlib, seaborn, scikit-learn, marimo.

Shared utilities live at the repo root in `utils/` (`eda_utils.py`, `model_utils.py`, `stats_utils.py`) and are imported via `sys.path.append("../../../utils")`.

## Project

**INN Hotels — Booking Cancellation Prediction** — binary classification on ~36k hotel-booking records to predict cancellations and inform refund and overbooking policies. Compares KNN, Naive Bayes, and SVM (with kernel and parameter tuning); selects a final model under a recall-optimised cost-of-errors framing.

- Problem statement: [project/problem_statement.md](project/problem_statement.md)
- Submission rubric: [project/submission_guidelines.md](project/submission_guidelines.md)
- Notebook: [project/inn_hotels.py](project/inn_hotels.py) (marimo)
- Reference notebooks (course-supplied, not the deliverable): [project/ML2_INN_Full_code_Learner_notebook.ipynb](project/ML2_INN_Full_code_Learner_notebook.ipynb), [project/ML2_INN_Low_Code_Learner_notebook.ipynb](project/ML2_INN_Low_Code_Learner_notebook.ipynb)
