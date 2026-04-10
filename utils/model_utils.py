"""Reusable model evaluation utilities.

Functions for classification model performance metrics and confusion matrix plots.
Framework-agnostic (sklearn-based). For framework-specific helpers (TensorFlow, etc.),
add them in the course project itself.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)


def classification_report_df(model, predictors, target):
    """Compute accuracy, recall, precision, and F1-score as a DataFrame.

    Works with sklearn-compatible models (model.predict returns class labels).
    """
    pred = model.predict(predictors)
    return pd.DataFrame(
        {
            "Accuracy": accuracy_score(target, pred),
            "Recall": recall_score(target, pred),
            "Precision": precision_score(target, pred),
            "F1": f1_score(target, pred),
        },
        index=[0],
    )


def classification_report_proba(model, predictors, target, threshold=0.5):
    """Compute classification metrics for models that return probabilities.

    Works with statsmodels or any model where predict() returns probabilities.
    """
    pred_probs = model.predict(predictors)
    pred = (pred_probs > threshold).astype(int)

    if hasattr(target, "to_numpy"):
        target_array = target.to_numpy().reshape(-1)
    else:
        target_array = np.array(target).reshape(-1)

    return pd.DataFrame(
        {
            "Accuracy": accuracy_score(target_array, pred),
            "Recall": recall_score(target_array, pred),
            "Precision": precision_score(target_array, pred),
            "F1": f1_score(target_array, pred),
        },
        index=[0],
    )


def plot_confusion_matrix(model, predictors, target):
    """Plot confusion matrix with percentages for an sklearn-compatible model."""
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [f"{item}\n{item / cm.sum():.2%}" for item in cm.flatten()]
    ).reshape(cm.shape)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")

    plt.show()
