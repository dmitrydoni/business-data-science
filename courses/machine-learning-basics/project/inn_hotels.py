import marimo

__generated_with = "0.23.0"
app = marimo.App(
    width="full",
    app_title="Machine Learning Basics: INN Hotels",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <p float="center"><center>
      <img src="https://walshcollege.edu/wp-content/uploads/2025/02/WALSH-COLLEGE-LOGO-WHITE-NO-TM.png.webp" width="200" height="100"/>
      <img src="https://mma.prnewswire.com/media/1458111/Great_Learning_Logo.jpg" width="200" height="100"/>
    </center></p>
    <h1><center>Course: Machine Learning Basics</center></h1>
    <h2><center>Project: INN Hotels — Booking Cancellation Prediction</center></h2>
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

    A significant number of hotel bookings are called off due to cancellations or no-shows. Typical reasons include change of plans and scheduling conflicts. The option to cancel free of charge or at a low cost is convenient for guests but revenue-diminishing for hotels — particularly on last-minute cancellations.

    Online booking channels have changed customers' booking possibilities and behavior, adding a further dimension to the cancellation challenge.

    Cancellations impact a hotel on several fronts:

    1. Loss of revenue when the room cannot be resold.
    2. Higher distribution-channel costs (commissions, last-minute publicity).
    3. Margin compression from last-minute price drops to resell.
    4. Operational overhead of rearranging guest accommodations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Objective

    INN Hotels Group operates hotels in Portugal and is facing high cancellation rates. As a data scientist, the task is to:

    1. Identify factors with the highest influence on booking cancellations.
    2. Build a predictive model that flags likely cancellations in advance.
    3. Translate findings into profitable cancellation- and refund-policy recommendations.

    The deliverable is a fully reproducible notebook exported to HTML, comparing KNN, Naive Bayes, and SVM (with kernel/parameter tuning) and selecting a final model with rationale.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Dictionary

    - `Booking_ID` — unique identifier of each booking *(dropped before modelling)*
    - `no_of_adults` — number of adults
    - `no_of_children` — number of children
    - `no_of_weekend_nights` — number of weekend nights (Sat/Sun) booked
    - `no_of_week_nights` — number of weeknights (Mon–Fri) booked
    - `type_of_meal_plan` — Not Selected · Meal Plan 1 (breakfast) · Meal Plan 2 (half board) · Meal Plan 3 (full board)
    - `required_car_parking_space` — 0 (no) / 1 (yes)
    - `room_type_reserved` — encoded (Room_Type 1…7)
    - `lead_time` — days between booking date and arrival date
    - `arrival_year`, `arrival_month`, `arrival_date` — arrival date components
    - `market_segment_type` — Online · Offline · Corporate · Aviation · Complementary
    - `repeated_guest` — 0 (no) / 1 (yes)
    - `no_of_previous_cancellations` — prior cancellations by this customer
    - `no_of_previous_bookings_not_canceled` — prior completed bookings
    - `avg_price_per_room` — average price per day in euros (dynamic)
    - `no_of_special_requests` — count of special requests
    - **`booking_status`** — target: Canceled / Not_Canceled
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Evaluation Criterion

    **Cost-of-errors framing.** A missed cancellation (FN — predicted not_cancelled, actually cancels) costs the hotel a room that could have been overbooked or resold; a false alarm (FP — predicted to cancel, actually shows up) costs at most a retention nudge or polite outreach. **FN is more expensive than FP**.

    → **Optimise for recall on the cancelled class.** Track F1 and the full confusion matrix as secondary indicators. Use AUC-ROC for threshold-independent comparison across models.

    Default threshold of 0.5 is unlikely to be optimal — sweep at the end and pick the operating point that meets a recall target with the highest precision.

    ***
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Libraries
    *Imports and personal utils*
    """)
    return


@app.cell
def _():
    import marimo as mo

    import sys
    from pathlib import Path
    sys.path.append("../../../utils")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        accuracy_score, recall_score, precision_score, f1_score,
        confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve,
        classification_report,
    )

    import warnings
    warnings.filterwarnings("ignore")

    from eda_utils import (
        histogram_boxplot,
        labeled_barplot,
        stacked_barplot,
        distribution_plot_wrt_target,
    )
    from model_utils import (
        classification_report_df,
        plot_confusion_matrix,
        classification_report_predict_proba,
        plot_confusion_matrix_proba,
        compare_models_table,
    )
    return (
        GaussianNB,
        GridSearchCV,
        KNeighborsClassifier,
        Path,
        SVC,
        StandardScaler,
        StratifiedKFold,
        accuracy_score,
        classification_report,
        classification_report_df,
        classification_report_predict_proba,
        compare_models_table,
        confusion_matrix,
        distribution_plot_wrt_target,
        f1_score,
        histogram_boxplot,
        labeled_barplot,
        mo,
        np,
        pd,
        permutation_importance,
        plot_confusion_matrix,
        plot_confusion_matrix_proba,
        plt,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
        sns,
        stacked_barplot,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the dataset
    """)
    return


@app.cell
def _(pd):
    hotel = pd.read_csv("data/raw/INNHotelsGroup.csv")
    data = hotel.copy()  # preserve original
    return data, hotel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Overview
    *Shape · dtypes · duplicates · sanity checks · drop ID*
    """)
    return


@app.cell
def _(data):
    data.head()


@app.cell
def _(data):
    data.tail()


@app.cell
def _(data):
    data.shape


@app.cell
def _(data):
    data.dtypes


@app.cell
def _(data):
    data.duplicated().sum()


@app.cell
def _(data):
    # Drop unique-identifier column — adds no signal
    data_m = data.drop(columns=["Booking_ID"])
    data_m.head()
    return (data_m,)


@app.cell
def _(data_m):
    data_m.describe().T


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploratory Data Analysis (EDA)

    ### EDA questions to answer

    1. What are the busiest months in the hotel?
    2. Which market segment do most guests come from?
    3. How do room prices vary across market segments?
    4. What percentage of bookings are canceled?
    5. What percentage of repeating guests cancel?
    6. Do special requests affect cancellation likelihood?

    *Plus deeper analysis beyond the seeded questions for full marks.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Univariate analysis
    *Distribution of each feature*
    """)
    return


@app.cell
def _(data_m, histogram_boxplot):
    # TODO: histogram_boxplot for each numeric: lead_time, avg_price_per_room,
    # no_of_adults, no_of_children, no_of_week_nights, no_of_weekend_nights,
    # no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
    # no_of_special_requests, arrival_month, arrival_date
    histogram_boxplot(data_m, "lead_time")


@app.cell
def _(data_m, labeled_barplot):
    # TODO: labeled_barplot for each categorical:
    # type_of_meal_plan, required_car_parking_space, room_type_reserved,
    # market_segment_type, repeated_guest, booking_status
    labeled_barplot(data_m, "booking_status", perc=True)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Encode target

    `Canceled` → 1, `Not_Canceled` → 0 — for downstream correlation and modelling.
    """)
    return


@app.cell
def _(data_m):
    data_m["booking_status"] = (data_m["booking_status"] == "Canceled").astype(int)
    data_m["booking_status"].value_counts(normalize=True)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bivariate analysis
    *Correlations · target-conditional distributions · seeded EDA questions*
    """)
    return


@app.cell
def _(data_m, np, plt, sns):
    # Correlation heatmap (numeric only)
    num_cols = data_m.select_dtypes(include=np.number).columns.tolist()
    plt.figure(figsize=(12, 7))
    sns.heatmap(data_m[num_cols].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
    plt.show()


@app.cell
def _():
    # TODO: market_segment_type vs avg_price_per_room (boxplot)
    # TODO: no_of_special_requests vs avg_price_per_room (boxplot, no outliers)
    # TODO: distribution_plot_wrt_target for avg_price_per_room vs booking_status
    # TODO: monthly arrival count line plot (busiest months)
    # TODO: stacked_barplot for market_segment_type vs booking_status (cancellation rate per segment)
    # TODO: stacked_barplot for repeated_guest vs booking_status
    # TODO: stacked_barplot for no_of_special_requests vs booking_status
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Preprocessing
    *Outlier inspection · feature engineering · scaling · encoding · split*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Outlier inspection

    Don't auto-cap. Reason per feature:

    - `avg_price_per_room` zeros — investigate (likely complementary segment); cap at upper IQR whisker only after confirming non-zero outliers are erroneous, not real luxury suites.
    - `lead_time` extremes — long-lead bookings are real; usually keep.
    - `no_of_previous_cancellations` / `_not_canceled` — heavy-tailed but legitimate behavior.
    """)
    return


@app.cell
def _():
    # TODO: numeric outlier visualisation (matplotlib boxplot grid, see learner notebook)
    # TODO: per-feature cap-or-keep decision with rationale
    # TODO: avg_price_per_room — investigate zeros; IQR upper-whisker cap if appropriate
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train-test split

    Stratified 80/20 — preserves cancellation rate in both folds.
    """)
    return


@app.cell
def _(data_m, train_test_split):
    X = data_m.drop(columns=["booking_status"])
    y = data_m["booking_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Train class %:", y_train.value_counts(normalize=True).to_dict())
    print("Test  class %:", y_test.value_counts(normalize=True).to_dict())
    return X, X_test, X_train, y, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Encoding categoricals + scaling numerics

    Pipeline pattern: fit on train only, transform both. `StandardScaler` for numeric (KNN and SVM are scale-sensitive); one-hot with `drop_first=True` for categoricals.
    """)
    return


@app.cell
def _():
    # TODO: build ColumnTransformer with StandardScaler(num_cols) + OneHotEncoder(drop='first', cat_cols)
    # Fit on X_train, transform X_train and X_test
    # Result: X_train_p, X_test_p as numpy arrays or DataFrames
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Building
    *Three baseline classifiers — KNN (k=3), Naive Bayes, SVM (linear)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### KNN — k=3 baseline
    """)
    return


@app.cell
def _():
    # TODO: knn_0 = KNeighborsClassifier(n_neighbors=3)
    # TODO: fit on X_train_p, y_train
    # TODO: plot_confusion_matrix(knn_0, X_train_p, y_train)
    # TODO: knn_0_train = classification_report_df(knn_0, X_train_p, y_train)
    # TODO: plot_confusion_matrix(knn_0, X_test_p, y_test)
    # TODO: knn_0_test = classification_report_df(knn_0, X_test_p, y_test)
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Naive Bayes — GaussianNB
    """)
    return


@app.cell
def _():
    # TODO: nb = GaussianNB()
    # TODO: fit, confusion matrix, train+test report
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SVM — linear kernel baseline
    """)
    return


@app.cell
def _():
    # TODO: svm_0 = SVC(kernel='linear', probability=True, random_state=42)
    # TODO: fit, confusion matrix, train+test report
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Performance Improvement
    *KNN k-sweep · SVM kernel × C × gamma sweep (≥ 5 combinations)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### KNN k-sweep
    """)
    return


@app.cell
def _():
    # TODO: sweep odd k in [3, 19], plot train and test recall (or F1) vs k, pick optimal
    # TODO: refit KNN at chosen k, full performance report
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SVM tuning — kernel and parameters

    At least 5 SVM combinations across `kernel ∈ {linear, poly d=2, poly d=3, rbf}`, `gamma ∈ {default, 0.016}`, `C ∈ {default, 0.1}`. See course's MLS-2 notebook for the canonical sweep order.
    """)
    return


@app.cell
def _():
    # TODO: SVM Polynomial Kernel, degree=2
    # TODO: SVM Polynomial Kernel, degree=2, gamma=0.016
    # TODO: SVM Polynomial Kernel, degree=2, gamma=0.016, C=0.1
    # TODO: SVM Polynomial Kernel, degree=3 (and variants)
    # TODO: SVM RBF Kernel (and variants with gamma, C)
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Performance Comparison
    *Side-by-side train and test metrics for all 13+ variants*
    """)
    return


@app.cell
def _():
    # TODO: build comparison DataFrame for all (knn_0, knn_tuned, nb, svm_linear,
    # svm_poly_d2, svm_poly_d2_gamma, svm_poly_d2_gamma_C, svm_poly_d3, svm_poly_d3_gamma,
    # svm_poly_d3_gamma_C, svm_rbf, svm_rbf_gamma, svm_rbf_gamma_C)
    # TODO: print train table, then test table
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Final model selection

    Pick the model with the **highest test recall** under an acceptable train-test gap. Tie-breakers: training cost (KNN has none, SVM-RBF has high cost on 36k rows), simplicity, deployment fit.
    """)
    return


@app.cell
def _():
    # TODO: name the chosen model, justify per criteria
    # best_estimator = ...
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Threshold tuning

    The default 0.5 threshold is rarely optimal under cost-asymmetric framing. Sweep the threshold on validation, plot precision-recall curve, pick the operating point matching the business constraint.
    """)
    return


@app.cell
def _():
    # TODO: precision_recall_curve(y_test, best_estimator.predict_proba(X_test_p)[:, 1])
    # TODO: pick threshold; report metrics at chosen threshold via classification_report_predict_proba
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Permutation feature importance

    Model-agnostic feature importance under the chosen metric — honest about what the *fitted* model relies on.
    """)
    return


@app.cell
def _():
    # TODO: permutation_importance(best_estimator, X_test_p, y_test, scoring='recall',
    #                              n_repeats=10, random_state=42, n_jobs=-1)
    # TODO: barplot of importance_mean per feature
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Actionable Insights and Recommendations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Insights

    *EDA + model findings translated into business observations.*

    - TODO
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Business Recommendations

    *Concrete actions for INN Hotels' cancellation and refund policies.*

    - TODO: tier the lead-time / market-segment / price interactions into actionable segments
    - TODO: refund-policy levers tied to predicted cancellation risk
    - TODO: overbooking strategy keyed on top-N highest-risk bookings
    - TODO: special-request and repeat-guest patterns and what they mean for retention

    ***
    """)
    return


if __name__ == "__main__":
    app.run()
