"""Reusable EDA visualization utilities.

Functions for common exploratory data analysis plots:
histogram + boxplot combos, labeled bar charts, stacked bar charts,
and distribution plots segmented by a target variable.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """Boxplot and histogram combined.

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )

    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="violet")

    if bins:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter")
    else:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2)

    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")

    plt.show()


def labeled_barplot(data, feature, figsize=(10, 5), perc=False, n=None):
    """Barplot with count or percentage labels at the top.

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default False)
    n: display only the top n category levels (default None = all)
    """
    total = len(data[feature])
    plt.figure(figsize=figsize)
    plt.xticks(rotation=90, fontsize=12)

    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc:
            label = "{:.1f}%".format(100 * p.get_height() / total)
        else:
            label = p.get_height()

        x = p.get_x() + p.get_width() / 2
        y = p.get_height()

        ax.annotate(
            label, (x, y), ha="center", va="center",
            size=12, xytext=(0, 5), textcoords="offset points",
        )

    plt.show()


def stacked_barplot(data, predictor, target):
    """Category counts and stacked bar chart.

    data: dataframe
    predictor: independent variable column name
    target: target variable column name
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]

    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)

    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.show()


def distribution_plot_wrt_target(data, predictor, target):
    """Distribution plots (histogram + boxplot) segmented by target classes."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    target_uniq = data[target].unique()

    axs[0, 0].set_title(f"Distribution for target={target_uniq[0]}")
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor, kde=True, ax=axs[0, 0], color="teal", stat="density",
    )

    axs[0, 1].set_title(f"Distribution for target={target_uniq[1]}")
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor, kde=True, ax=axs[0, 1], color="orange", stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data, x=target, y=predictor, ax=axs[1, 1],
        showfliers=False, palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()
