import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full", app_title="E-news Express: A/B Testing")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # E-news Express: A/B Testing

    **Course:** Business Statistics
    **Author:** Dmitry Doni
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem Statement

    E-news Express conducted an A/B test comparing the old landing page (control) with a new landing page (treatment) across 100 randomly selected users. The goal is to determine, at a 5% significance level, whether the new page is more effective at converting visitors into subscribers.

    **Questions:**

    1. Do users spend more time on the new landing page than on the old one?
    2. Is the conversion rate for the new page greater than for the old page?
    3. Does conversion status depend on the preferred language?
    4. Is the time spent on the new page the same across different language users?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the Data
    """)
    return


@app.cell
def _(pd):
    from pathlib import Path

    _project_dir = Path(__file__).resolve().parent
    df = pd.read_csv(_project_dir / "data" / "raw" / "abtest.csv")
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploratory Data Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Overview
    """)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Univariate Analysis
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bivariate Analysis
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Do users spend more time on the new landing page than on the old one?

    ### Visual Analysis
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hypotheses

    - $H_0$:
    - $H_1$:

    ### Test Selection

    ### Significance Level

    $\alpha = 0.05$
    """)
    return


@app.cell
def _():
    # Calculate p-value
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inference
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Is the conversion rate for the new page greater than for the old page?

    ### Visual Analysis
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hypotheses

    - $H_0$:
    - $H_1$:

    ### Test Selection

    ### Significance Level

    $\alpha = 0.05$
    """)
    return


@app.cell
def _():
    # Calculate p-value
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inference
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Does conversion status depend on the preferred language?

    ### Visual Analysis
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hypotheses

    - $H_0$:
    - $H_1$:

    ### Test Selection

    ### Significance Level

    $\alpha = 0.05$
    """)
    return


@app.cell
def _():
    # Calculate p-value
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inference
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Is the time spent on the new page the same across different language users?

    ### Visual Analysis
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hypotheses

    - $H_0$:
    - $H_1$:

    ### Test Selection

    ### Significance Level

    $\alpha = 0.05$
    """)
    return


@app.cell
def _():
    # Calculate p-value
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inference
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion and Business Recommendations
    """)
    return


if __name__ == "__main__":
    app.run()
