"""Reusable statistical visualization utilities.

Functions for visual analysis in hypothesis testing:
group comparison boxplots, proportion comparison bar charts.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def compare_groups_boxplot(
    data,
    x,
    y,
    title=None,
    xlabel=None,
    ylabel=None,
    figsize=(8, 5),
    palette="coolwarm",
    print_stats=True,
):
    """Boxplot comparing a numeric variable across groups, with means shown.

    data: dataframe
    x: categorical column (group variable)
    y: numeric column (variable to compare)
    title: plot title (default: auto-generated)
    xlabel: x-axis label (default: column name)
    ylabel: y-axis label (default: column name)
    figsize: size of figure (default (8, 5))
    palette: seaborn color palette (default "coolwarm")
    print_stats: whether to print group mean, std, count (default True)
    """
    order = sorted(data[x].unique())

    plt.figure(figsize=figsize)
    sns.boxplot(
        x=x, y=y, data=data, hue=x,
        showmeans=True, palette=palette,
        order=order, hue_order=order, dodge=False, legend=False,
    )
    plt.title(title or f"{y} by {x}")
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.show()

    if print_stats:
        print(data.groupby(x)[y].agg(["mean", "std", "count"]))


def compare_proportions_bar(
    data,
    x,
    y,
    title=None,
    xlabel=None,
    ylabel="Proportion",
    figsize=(8, 5),
    colormap="coolwarm",
):
    """Stacked bar chart comparing proportions of a binary variable across groups.

    data: dataframe
    x: categorical column (group variable)
    y: categorical column (outcome variable, e.g. converted)
    title: plot title (default: auto-generated)
    xlabel: x-axis label (default: column name)
    ylabel: y-axis label (default "Proportion")
    figsize: size of figure (default (8, 5))
    colormap: matplotlib colormap name (default "coolwarm")
    """
    pd.crosstab(data[x], data[y], normalize="index").plot(
        kind="bar", stacked=True, figsize=figsize, colormap=colormap,
    )
    plt.title(title or f"{y} by {x}")
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel)
    plt.legend(title=y)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
