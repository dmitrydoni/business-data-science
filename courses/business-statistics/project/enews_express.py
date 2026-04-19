import marimo

__generated_with = "0.23.0"
app = marimo.App(
    width="full",
    app_title="Business Statistics: E-news Express",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <p float="center"><center>
      <img src="https://walshcollege.edu/wp-content/uploads/2025/02/WALSH-COLLEGE-LOGO-WHITE-NO-TM.png.webp" width="200" height="100"/>
      <img src="https://mma.prnewswire.com/media/1458111/Great_Learning_Logo.jpg" width="200" height="100"/>
    </center></p>
    <h1><center>Course: Business Statistics</center></h1>
    <h2><center>Project: E-News Express</center></h2>
    <h2><center>Author: Dmitry Doni</center></h2>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Problem Statement
    *Context, objective, data dictionary*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Context

    The advent of e-news, or electronic news, portals has offered us a great opportunity to quickly get updates on the day-to-day events occurring globally. The information on these portals is retrieved electronically from online databases, processed using a variety of software, and then transmitted to the users. There are multiple advantages of transmitting news electronically, like faster access to the content and the ability to utilize different technologies such as audio, graphics, video, and other interactive elements that are either not being used or aren't common yet in traditional newspapers.

    E-News Express, an online news portal, aims to expand its business by acquiring new subscribers. With every visitor to the website taking certain actions based on their interest, the company plans to analyze these actions to understand user interests and determine how to drive better engagement. The executives at E-News Express are of the opinion that there has been a decline in new monthly subscribers compared to the past year because the current webpage is not designed well enough in terms of the outline and recommended content to keep customers engaged long enough to make a decision to subscribe.

    Companies often analyze user responses to two variants of a product to decide which of the two variants is more effective. This experimental technique, known as **A/B testing**, is used to determine whether a new feature attracts users based on a chosen metric.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Objective

    The design team of the company has researched and created a new landing page that has a new outline and more relevant content shown compared to the old page. In order to test the effectiveness of the new landing page in gathering new subscribers, the Data Science team conducted an experiment by randomly selecting 100 users and dividing them equally into two groups. The existing landing page was served to the first group (control group) and the new landing page to the second group (treatment group). Data regarding the interaction of users in both groups with the two versions of the landing page was collected.

    Being a data scientist in E-News Express, you have been asked to explore the data and perform a statistical analysis (at a significance level of 5%) to determine the effectiveness of the new landing page in gathering new subscribers for the news portal by answering the following questions:

    1. Do the users spend more time on the new landing page than on the existing landing page?
    2. Is the conversion rate (the proportion of users who visit the landing page and get converted) for the new page greater than the conversion rate for the old page?
    3. Does the converted status depend on the preferred language?
    4. Is the time spent on the new page the same for the different language users?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Dictionary

    - `user_id`: Unique user ID of the person visiting the website
    - `group`: Incidates whether the user belongs to the first group (control) or the second group (treatment)
    - `landing_page`: Indicates whether the landing page is new or old
    - `time_spent_on_the_page`: Time (in minutes) spent by the user on the landing page
    - `converted`: Indicates whether the user gets converted to a subscriber of the news portal or not
    - `language_preferred`: Language chosen by the user to view the landing page

    ***
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Libraries
    *Libraries used in the project*
    """)
    return


@app.cell
def _():
    import marimo as mo

    import sys
    from pathlib import Path
    sys.path.append("../../utils")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from scipy import stats
    from scipy.stats import ttest_ind, chi2_contingency, f_oneway
    from statsmodels.stats.proportion import proportions_ztest

    import warnings
    warnings.filterwarnings("ignore")

    # Personal utils
    from eda_utils import histogram_boxplot, labeled_barplot, stacked_barplot, distribution_plot_wrt_target
    from stats_utils import compare_groups_boxplot, compare_proportions_bar

    return (
        Path,
        chi2_contingency,
        compare_groups_boxplot,
        compare_proportions_bar,
        distribution_plot_wrt_target,
        f_oneway,
        histogram_boxplot,
        labeled_barplot,
        mo,
        pd,
        proportions_ztest,
        stacked_barplot,
        stats,
        ttest_ind,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ***
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploratory Data Analysis
    *Data overview, univariate analysis, bivariate analysis*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Loading Data
    """)
    return


@app.cell
def _(Path, pd):
    _project_dir = Path(__file__).resolve().parent

    df = pd.read_csv(_project_dir / "data" / "raw" / "abtest.csv")
    df.shape
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Overview
    *Observations and sanity checks*
    """)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.head(3)
    return


@app.cell
def _(df):
    df.tail(3)
    return


@app.cell
def _(df):
    df.describe(include="all")
    return


@app.cell
def _(df):
    # Check for missing values
    df.isnull().sum()
    return


@app.cell
def _(df):
    # Check for duplicates
    df.duplicated().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Observations

    - The dataset contains 100 observations and 6 features, no missing values and no duplicates.
    - The sample is evenly split: 50 control (old page) and 50 treatment (new page).
    - Features:
        - `time_spent_on_the_page` is the only numerical feature: mean 5.4 min, range 0.19–10.71 min, with moderate spread (std 2.4 min).
        - `group`, `landing_page`, `converted` categorical features are binary.
        - `language_preferred` has three levels: English, French, Spanish.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ***
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Univariate Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Distribution of time spent on the landing page
    """)
    return


@app.cell
def _(df, histogram_boxplot):
    histogram_boxplot(df, "time_spent_on_the_page", figsize=(6, 4), kde=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    - The distribution of time spent is roughly uniform, not strongly skewed.
    - Mean and median are close (5.4 min), confirming near-symmetric spread.
    - No significant outliers visible in the boxplot.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Distribution of users across groups
    """)
    return


@app.cell
def _(df, labeled_barplot):
    labeled_barplot(df, "group", figsize=(6, 4), perc=True)
    return


@app.cell
def _(df, labeled_barplot):
    labeled_barplot(df, "landing_page", figsize=(6, 4), perc=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    - The groups are perfectly balanced: 50 control / 50 treatment, matching old / new pages.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Overall conversion rate and language distribution
    """)
    return


@app.cell
def _(df, labeled_barplot):
    labeled_barplot(df, "converted", figsize=(6, 4), perc=True)
    return


@app.cell
def _(df, labeled_barplot):
    labeled_barplot(df, "language_preferred", figsize=(6, 4), perc=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    - Overall conversion rate: more than a half of the users converted (54%).
    - Language distribution is roughly balanced across English, French, and Spanish.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ***
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bivariate Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Time spent on the page: old vs. new landing page
    """)
    return


@app.cell
def _(df, distribution_plot_wrt_target):
    distribution_plot_wrt_target(df, "time_spent_on_the_page", "landing_page", figsize=(10, 6))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    - Users on the **new page** appear to spend more time on average compared to the old page.
    - The median and mean are both visibly higher for the new page.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Time spent on the page: conversion status
    """)
    return


@app.cell
def _(df, distribution_plot_wrt_target):
    distribution_plot_wrt_target(df, "time_spent_on_the_page", "converted", figsize=(10, 6))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    - Converted users tend to spend more time on the page than non-converted users.
    - This suggests that engagement (time spent) is positively associated with conversion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Conversion rate: old vs. new landing page
    """)
    return


@app.cell
def _(df, stacked_barplot):
    stacked_barplot(df, "landing_page", "converted", figsize=(4, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    - The new landing page has a visibly **higher conversion rate** compared to the old page.
    - This provides an initial visual indication that the new page may be more effective.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Conversion rate across languages
    """)
    return


@app.cell
def _(df, stacked_barplot):
    stacked_barplot(df, "language_preferred", "converted", figsize=(4, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    - Conversion rates appear relatively similar across all three languages.
    - Visual evidence shows English users having slightly higher conversion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Time spent on the new page across language groups
    """)
    return


@app.cell
def _(df):
    df_new = df[df["landing_page"] == "new"]
    return (df_new,)


@app.cell
def _(df_new, distribution_plot_wrt_target):
    distribution_plot_wrt_target(df_new, "time_spent_on_the_page", "language_preferred", figsize=(12, 6))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    - Different language groups show similar patterns; fewer Spanish users spend more than 8 minutes on the new page.
    - The boxplot shows comparable medians and IQRs across all three language groups.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ***
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hypothesis Testing

    *Visual analysis, hypotheses, test selection, inference*
    """)
    return


@app.cell
def _():
    ALPHA = 0.05  # significance level 5%
    return (ALPHA,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Do users spend more time on the new landing page than on the old one?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Visual Analysis
    """)
    return


@app.cell
def _(compare_groups_boxplot, df):
    compare_groups_boxplot(
        df, x="landing_page", y="time_spent_on_the_page",
        title="Time Spent on the Page by Landing Page Version",
        xlabel="Landing Page", ylabel="Time Spent (minutes)",
        figsize=(8, 4)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hypotheses

    Let $\mu_{new}$ be the mean time spent on the new page and $\mu_{old}$ be the mean time spent on the old page.

    - $H_0: \mu_{new} \leq \mu_{old}$ (Users do not spend more time on the new page)
    - $H_a: \mu_{new} > \mu_{old}$ (Users spend more time on the new page)

    #### Test Selection

    This is a **one-tailed test** comparing means from **two independent populations** (control vs. treatment).
    The population standard deviations are unknown and estimated from the samples.
    Per the Hypothesis Testing Framework, this calls for a **two-sample independent t-test**.

    We check the sample standard deviations to decide whether to assume equal variances.
    """)
    return


@app.cell
def _(df):
    # Compare sample standard deviations
    time_new = df[df["landing_page"] == "new"]["time_spent_on_the_page"]
    time_old = df[df["landing_page"] == "old"]["time_spent_on_the_page"]

    print(f"Std (new): {time_new.std():.2f}")
    print(f"Std (old): {time_old.std():.2f}")
    return time_new, time_old


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The sample standard deviations differ noticeably (1.82 vs. 2.58). We use **Levene's test** to formally check the equality of variances and set `equal_var` accordingly.

    #### Significance Level

    $\alpha = 0.05$
    """)
    return


@app.cell
def _(stats, time_new, time_old):
    # Levene's test for equality of variances
    levene_stat_1, levene_p_1 = stats.levene(time_new, time_old)
    print(f"\nLevene's test:\nstatistic={levene_stat_1:.4f}, p-value={levene_p_1:.4f}")
    if levene_p_1 > 0.05:
        print("Equal variances assumption holds (equal_var=True)")
    else:
        print("Variances are unequal, use Welch's t-test (equal_var=False)")
    return (levene_p_1,)


@app.cell
def _(levene_p_1, time_new, time_old, ttest_ind):
    # Two-sample independent t-test (one-tailed: new > old)
    # Using Welch's t-test if Levene's test rejects equal variances
    equal_var = levene_p_1 > 0.05
    test_stat_1, p_value_1 = ttest_ind(time_new, time_old, equal_var=equal_var, alternative="greater")

    print(f"Equal variances: {equal_var}")
    print(f"Test statistic: {test_stat_1:.4f}")
    print(f"P-value: {p_value_1:.6f}")
    return (p_value_1,)


@app.cell
def _(ALPHA, p_value_1):
    # Compare p-value with alpha
    if p_value_1 < ALPHA:
        print(f"As the p-value ({p_value_1:.6f}) is less than the level of significance ({ALPHA}), we reject the null hypothesis.")
    else:
        print(f"As the p-value ({p_value_1:.6f}) is greater than the level of significance ({ALPHA}), we fail to reject the null hypothesis.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Inference

    At a 5% significance level, we reject the null hypothesis. There is sufficient statistical evidence to conclude that users spend **more time on the new landing page** than on the old one. The new page design appears to drive higher engagement.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ***
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Is the conversion rate for the new page greater than for the old page?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Visual Analysis
    """)
    return


@app.cell
def _(compare_proportions_bar, df):
    compare_proportions_bar(
        df, x="group", y="converted",
        title="Conversion Rate by Group",
        xlabel="Group",
        figsize=(6, 4)
    )
    return


@app.cell
def _(df):
    # Prepare data: counts and totals for each group
    new_converted = df[df["group"] == "treatment"]["converted"].value_counts()["yes"]
    old_converted = df[df["group"] == "control"]["converted"].value_counts()["yes"]
    n_treatment = df["group"].value_counts()["treatment"]
    n_control = df["group"].value_counts()["control"]

    print(f"Treatment: {new_converted} converted out of {n_treatment} (rate: {new_converted / n_treatment:.2%})")
    print(f"Control:   {old_converted} converted out of {n_control} (rate: {old_converted / n_control:.2%})")
    return n_control, n_treatment, new_converted, old_converted


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hypotheses

    Let $p_{new}$ be the conversion rate for the new page and $p_{old}$ be the conversion rate for the old page.

    - $H_0: p_{new} \leq p_{old}$ (The conversion rate for the new page is not greater)
    - $H_a: p_{new} > p_{old}$ (The conversion rate for the new page is greater)

    #### Test Selection

    This is a **one-tailed test** comparing **two population proportions** from independent groups.
    Per the Hypothesis Testing Framework, this calls for a **two-proportion z-test**.

    #### Significance Level

    $\alpha = 0.05$
    """)
    return


@app.cell
def _(n_control, n_treatment, new_converted, old_converted, proportions_ztest):
    # Two-proportion z-test (one-tailed: new > old)
    test_stat_2, p_value_2 = proportions_ztest(
        [new_converted, old_converted],
        [n_treatment, n_control],
        alternative="larger",
    )
    print(f"Test statistic: {test_stat_2:.4f}")
    print(f"P-value: {p_value_2:.6f}")
    return (p_value_2,)


@app.cell
def _(ALPHA, p_value_2):
    # Compare p-value with alpha
    if p_value_2 < ALPHA:
        print(f"As the p-value ({p_value_2:.6f}) is less than the level of significance ({ALPHA}), we reject the null hypothesis.")
    else:
        print(f"As the p-value ({p_value_2:.6f}) is greater than the level of significance ({ALPHA}), we fail to reject the null hypothesis.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Inference

    At a 5% significance level, we reject the null hypothesis. There is sufficient statistical evidence to conclude that the **conversion rate for the new page is greater** than for the old page. The new landing page design is more effective at converting visitors into subscribers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ***
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Does conversion status depend on the preferred language?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Visual Analysis
    """)
    return


@app.cell
def _(compare_proportions_bar, df):
    compare_proportions_bar(
        df, x="language_preferred", y="converted",
        title="Conversion Rate by Language Preferred",
        xlabel="Language Preferred", colormap="Set2",
        figsize=(6, 4)
    )
    return


@app.cell
def _(df, pd):
    # Create the contingency table
    contingency_table = pd.crosstab(df["converted"], df["language_preferred"])
    print(contingency_table)
    return (contingency_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hypotheses

    - $H_0$: Conversion status and preferred language are **independent**
    - $H_a$: Conversion status and preferred language are **not independent**

    #### Test Selection

    Both variables are categorical. Per the Hypothesis Testing Framework, this calls for a
    **Chi-square test of independence**.

    We use `chi2_contingency()` from `scipy.stats` on the contingency table created via `pd.crosstab()`.

    #### Significance Level

    $\alpha = 0.05$
    """)
    return


@app.cell
def _(chi2_contingency, contingency_table):
    # Chi-square test of independence
    chi2, p_value_3, dof, expected_freq = chi2_contingency(contingency_table)

    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value_3:.6f}")
    print(f"\nExpected frequencies:\n{expected_freq}")
    return (p_value_3,)


@app.cell
def _(ALPHA, p_value_3):
    # Compare p-value with alpha
    if p_value_3 < ALPHA:
        print(f"As the p-value ({p_value_3:.6f}) is less than the level of significance ({ALPHA}), we reject the null hypothesis.")
    else:
        print(f"As the p-value ({p_value_3:.6f}) is greater than the level of significance ({ALPHA}), we fail to reject the null hypothesis.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Inference

    At a 5% significance level, we fail to reject the null hypothesis. There is **not enough statistical evidence** to say that conversion status depends on the preferred language. The conversion behavior appears to be independent of which language the user prefers.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ***
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. Is the time spent on the new page the same across different language users?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Visual Analysis
    """)
    return


@app.cell
def _(compare_groups_boxplot, df_new):
    compare_groups_boxplot(
        df_new, x="language_preferred", y="time_spent_on_the_page",
        title="Time Spent on New Page by Language Preferred",
        xlabel="Language Preferred", ylabel="Time Spent (minutes)",
        palette="pastel",
        figsize=(8, 4)
    )
    return


@app.cell
def _(df_new):
    # Prepare data: subset by language for treatment group only
    time_english = df_new[df_new["language_preferred"] == "English"]["time_spent_on_the_page"]
    time_french = df_new[df_new["language_preferred"] == "French"]["time_spent_on_the_page"]
    time_spanish = df_new[df_new["language_preferred"] == "Spanish"]["time_spent_on_the_page"]
    return time_english, time_french, time_spanish


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Hypotheses

    Let $\mu_E$, $\mu_F$, $\mu_S$ be the mean time spent on the new page by English, French, and Spanish users respectively.

    - $H_0: \mu_E = \mu_F = \mu_S$ (Mean time spent is the same across all language groups)
    - $H_a$: At least one of the means is different

    #### Test Selection

    This is a problem concerning **three population means**. One-way ANOVA
    is appropriate provided the assumptions of **normality** and **equality of variances** are satisfied.
    We verify both using Shapiro-Wilk's test and Levene's test.

    #### Significance Level

    $\alpha = 0.05$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Assumption Checks
    """)
    return


@app.cell
def _(stats, time_english, time_french, time_spanish):
    # Assumption 1: Normality (Shapiro-Wilk test for each group)
    for name, group in [("English", time_english), ("French", time_french), ("Spanish", time_spanish)]:
        w, p = stats.shapiro(group)

        print(f"Shapiro-Wilk test for {name}: W={w:.4f}, p-value={p:.4f} | {'Normal' if p > 0.05 else 'Not Normal'}")
    return


@app.cell
def _(stats, time_english, time_french, time_spanish):
    # Assumption 2: Homogeneity of Variance (Levene's test)
    levene_stat, levene_p = stats.levene(time_english, time_french, time_spanish)

    print(f"Levene's test: statistic={levene_stat:.4f}, p-value={levene_p:.4f}")

    if levene_p > 0.05:
        print("Equal variances assumption holds. We should proceed with ANOVA.")
    else:
        print("Variances are unequal. We should consider Kruskal-Wallis.")
    return


@app.cell
def _(f_oneway, time_english, time_french, time_spanish):
    # One-way ANOVA
    test_stat_4, p_value_4 = f_oneway(time_english, time_french, time_spanish)

    print(f"F-statistic: {test_stat_4:.4f}")
    print(f"P-value: {p_value_4:.6f}")
    return (p_value_4,)


@app.cell
def _(ALPHA, p_value_4):
    # Compare p-value with alpha
    if p_value_4 < ALPHA:
        print(f"As the p-value ({p_value_4:.6f}) is less than the level of significance ({ALPHA}), we reject the null hypothesis.")
    else:
        print(f"As the p-value ({p_value_4:.6f}) is greater than the level of significance ({ALPHA}), we fail to reject the null hypothesis.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Inference

    At a 5% significance level, we fail to reject the null hypothesis. There is **not enough statistical evidence** to say that the mean time spent on the new page differs across language groups. The new page engages users equally regardless of their preferred language.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ***
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary


    1. **Do users spend more time on the new page?**
    - Test: Two-sample t-test
    - Decision: **Reject $H_0$**
    - Answer: **Yes, significantly more time**

    2. **Is the conversion rate higher for the new page?**
    - Test: Two-proportion z-test
    - Decision: **Reject $H_0$**
    - Answer: **Yes, significantly higher**

    3. **Does conversion depend on language?**
    - Test: Chi-square test of independence
    - Decision: **Fail to reject $H_0$**
    - Answer: **No evidence of dependency**

    4. **Is time on new page the same across languages?**
    - Test: One-way ANOVA
    - Decision: **Fail to reject $H_0$**
    - Answer: **No significant difference**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Business Recommendations

    1. Deploy the new landing page. Both key metrics (time spent on the page, conversion rate) show statistically significant improvement over the old page. The new design drives higher engagement and converts more visitors into subscribers.

    2. No language-specific customization needed at this stage. Conversion rates and engagement levels are consistent across English, French, and Spanish users.

    3. Investigate engagement further. The bivariate analysis showed that users who convert tend to spend more time on the page. E-News Express could explore additional engagement strategies to further increase time on page and conversion rates.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
