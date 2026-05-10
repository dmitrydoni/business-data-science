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
    <h1><center>Course: Machine Learning II</center></h1>
    <h2><center>Project: INN Hotels - Booking Cancellation Prediction</center></h2>
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

    A significant number of hotel bookings are called off due to cancellations or no-shows. Typical reasons include change of plans and scheduling conflicts. The option to cancel free of charge or at a low cost is convenient for guests but revenue-diminishing for hotels, particularly on last-minute cancellations.

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

    Deliverable: data app comparing KNN, Naive Bayes, and SVM (with kernel/parameter tuning) and selecting a final model with rationale.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Data Dictionary

    - `Booking_ID`: unique identifier of each booking *(dropped before modelling)*
    - `no_of_adults`: number of adults
    - `no_of_children`: number of children
    - `no_of_weekend_nights`: number of weekend nights (Sat/Sun) booked
    - `no_of_week_nights`: number of weeknights (Mon–Fri) booked
    - `type_of_meal_plan`: Not Selected / Meal Plan 1 (breakfast) / Meal Plan 2 (half board) / Meal Plan 3 (full board)
    - `required_car_parking_space`: 0 (no) / 1 (yes)
    - `room_type_reserved`: encoded (Room_Type 1…7)
    - `lead_time`: days between booking date and arrival date
    - `arrival_year`, `arrival_month`, `arrival_date`: arrival date components
    - `market_segment_type`: Online / Offline / Corporate / Aviation / Complementary
    - `repeated_guest`: 0 (no) / 1 (yes)
    - `no_of_previous_cancellations`: prior cancellations by this customer
    - `no_of_previous_bookings_not_canceled`: prior completed bookings
    - `avg_price_per_room`: average price per day in euros (dynamic)
    - `no_of_special_requests`: count of special requests
    - **`booking_status`**: target - Canceled / Not_Canceled
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Evaluation Criterion

    **Cost-of-errors framing.** A missed cancellation (FN - predicted `not_cancelled`, actually cancels) costs the hotel a room that could have been overbooked or resold
    a false alarm (FP - predicted to cancel, actually shows up) costs at most a retention nudge or polite outreach. FN is more expensive than FP.

    **Optimise for recall on the cancelled class.** Track F1 and the full confusion matrix as secondary indicators. Use AUC-ROC for threshold-independent comparison across models.

    Default threshold of 0.5 is unlikely to be optimal, sweep at the end and pick the operating point that meets a recall target with the highest precision.

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

    sys.path.append(str(Path(__file__).resolve().parents[3] / "utils"))

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        recall_score,
        f1_score,
        precision_recall_curve,
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
    )

    return (
        ColumnTransformer,
        GaussianNB,
        KNeighborsClassifier,
        OneHotEncoder,
        Path,
        SVC,
        StandardScaler,
        classification_report_df,
        classification_report_predict_proba,
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
        recall_score,
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
def _(Path, pd):
    # CWD-independent path resolution
    DATA_PATH = Path(__file__).resolve().parent / "data" / "raw" / "INNHotelsGroup.csv"
    hotel = pd.read_csv(DATA_PATH)
    data = hotel.copy()  # preserve original
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Overview
    *Shape, dtypes, duplicates, sanity checks, drop ID*
    """)
    return


@app.cell
def _(data):
    data.head(3)
    return


@app.cell
def _(data):
    data.tail(3)
    return


@app.cell
def _(data):
    data.shape
    return


@app.cell
def _(data):
    data.dtypes
    return


@app.cell
def _(data):
    data.duplicated().sum()
    return


@app.cell
def _(data):
    # Drop unique-identifier column (adds no signal)
    data_m = data.drop(columns=["Booking_ID"])
    data_m.head()
    return (data_m,)


@app.cell
def _(data_m):
    data_m.describe().T
    return


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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Univariate analysis
    *Distribution of each feature*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `lead_time`: days between booking and arrival
    """)
    return


@app.cell
def _(data_m, histogram_boxplot):
    histogram_boxplot(data_m, "lead_time", figsize=(8, 4), kde=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `avg_price_per_room`: average daily room rate (EUR)
    """)
    return


@app.cell
def _(data_m, histogram_boxplot):
    histogram_boxplot(data_m, "avg_price_per_room", figsize=(8, 4), kde=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `no_of_adults`
    """)
    return


@app.cell
def _(data_m, labeled_barplot):
    labeled_barplot(data_m, "no_of_adults", figsize=(8, 4), perc=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `no_of_children`
    """)
    return


@app.cell
def _(data_m, labeled_barplot):
    labeled_barplot(data_m, "no_of_children", figsize=(8, 4), perc=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `no_of_week_nights` and `no_of_weekend_nights`
    """)
    return


@app.cell
def _(data_m, labeled_barplot):
    labeled_barplot(data_m, "no_of_week_nights", figsize=(10, 4), perc=True)
    return


@app.cell
def _(data_m, labeled_barplot):
    labeled_barplot(data_m, "no_of_weekend_nights", figsize=(8, 4), perc=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `no_of_previous_cancellations` and `no_of_previous_bookings_not_canceled`
    """)
    return


@app.cell
def _(data_m, histogram_boxplot):
    histogram_boxplot(data_m, "no_of_previous_cancellations", figsize=(8, 3))
    return


@app.cell
def _(data_m, histogram_boxplot):
    histogram_boxplot(data_m, "no_of_previous_bookings_not_canceled", figsize=(8, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `no_of_special_requests`
    """)
    return


@app.cell
def _(data_m, labeled_barplot):
    labeled_barplot(data_m, "no_of_special_requests", figsize=(8, 4), perc=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `arrival_month` and `arrival_date`
    """)
    return


@app.cell
def _(data_m, labeled_barplot):
    labeled_barplot(data_m, "arrival_month", figsize=(10, 4), perc=True)
    return


@app.cell
def _(data_m, labeled_barplot):
    labeled_barplot(data_m, "arrival_date", figsize=(14, 4), perc=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Categoricals

    Meal plan, parking, room type, market segment, repeated guest
    """)
    return


@app.cell
def _(data_m, labeled_barplot):
    labeled_barplot(data_m, "type_of_meal_plan", figsize=(8, 4), perc=True)
    return


@app.cell
def _(data_m, labeled_barplot):
    labeled_barplot(data_m, "required_car_parking_space", figsize=(6, 4), perc=True)
    return


@app.cell
def _(data_m, labeled_barplot):
    labeled_barplot(data_m, "room_type_reserved", figsize=(8, 4), perc=True)
    return


@app.cell
def _(data_m, labeled_barplot):
    labeled_barplot(data_m, "market_segment_type", figsize=(8, 4), perc=True)
    return


@app.cell
def _(data_m, labeled_barplot):
    labeled_barplot(data_m, "repeated_guest", figsize=(6, 3), perc=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Target: `booking_status`
    """)
    return


@app.cell
def _(data_m, labeled_barplot):
    labeled_barplot(data_m, "booking_status", figsize=(6, 3), perc=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Univariate observations

    **Numeric features:**

    - `lead_time` is right-skewed with a long tail; long-lead bookings are domain-legitimate.
    - `avg_price_per_room` is roughly normal around 100 EUR, with a zero-mass cluster (Complementary segment) and an upper tail (IQR-capped in preprocessing).
    - `no_of_previous_cancellations` and `no_of_previous_bookings_not_canceled` are zero-heavy: ~97% first-time guests.
    - `no_of_special_requests` concentrates at 0–2; 3+ is rare.
    - `arrival_month` peaks Aug–Oct. `arrival_date` is roughly uniform.

    **Categorical features:**

    - `type_of_meal_plan`: Meal Plan 1 dominates (76.7%), Not Selected (14.1%), Meal Plan 2 (9.1%); Meal Plan 3 negligible.
    - `required_car_parking_space`: only 3.1% of bookings request parking.
    - `room_type_reserved`: Room_Type 1 dominates (77.5%), Room_Type 4 (16.7%); other types are minorities.
    - `market_segment_type`: Online 64%, Offline 29%, Corporate 5.6%, Complementary 1%, Aviation <1%.
    - `repeated_guest`: 2.6% returning, 97.4% first-time.
    - **Target `booking_status`**: 67.2% Not_Canceled / 32.8% Canceled. Moderate imbalance.
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
    ### Encode target

    `Canceled`=1, `Not_Canceled`=0 (for downstream correlation and modelling)
    """)
    return


@app.cell
def _(data_m):
    data_m["booking_status"] = (data_m["booking_status"] == "Canceled").astype(int)
    data_m["booking_status"].value_counts(normalize=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bivariate analysis
    *Correlations, target-conditional distributions, seeded EDA questions*
    """)
    return


@app.cell
def _(data_m, np, plt, sns):
    # Correlation heatmap (numeric only)
    _corr_cols = data_m.select_dtypes(include=np.number).columns.tolist()
    plt.figure(figsize=(12, 7))
    sns.heatmap(data_m[_corr_cols].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### EDA Q1: Busiest months in the hotel
    """)
    return


@app.cell
def _(data_m, plt, sns):
    monthly = data_m.groupby("arrival_month").size().reset_index(name="bookings")
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=monthly, x="arrival_month", y="bookings", marker="o")
    plt.xticks(range(1, 13))
    plt.title("Bookings by arrival month")
    plt.xlabel("Arrival month")
    plt.ylabel("Number of bookings")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### EDA Q2: Market segment distribution
    """)
    return


@app.cell
def _(data_m, labeled_barplot):
    labeled_barplot(data_m, "market_segment_type", figsize=(8, 4), perc=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### EDA Q3: Room price by market segment
    """)
    return


@app.cell
def _(data_m, plt, sns):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data_m, x="market_segment_type", y="avg_price_per_room")
    plt.title("Average price per room by market segment")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### EDA Q4: Overall cancellation rate
    """)
    return


@app.cell
def _(data_m):
    rate = data_m["booking_status"].mean()
    print(f"Cancellation rate: {rate:.2%} ({int(rate * len(data_m)):,} of {len(data_m):,} bookings)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### EDA Q5: Cancellation rate by repeated-guest status
    """)
    return


@app.cell
def _(data_m, stacked_barplot):
    stacked_barplot(data_m, "repeated_guest", "booking_status", figsize=(6, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### EDA Q6: Cancellation rate by special requests
    """)
    return


@app.cell
def _(data_m, stacked_barplot):
    stacked_barplot(data_m, "no_of_special_requests", "booking_status", figsize=(8, 4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Cancellation rate by market segment
    """)
    return


@app.cell
def _(data_m, stacked_barplot):
    stacked_barplot(data_m, "market_segment_type", "booking_status", figsize=(6, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Cancellation rate by meal plan and room type
    """)
    return


@app.cell
def _(data_m, stacked_barplot):
    stacked_barplot(data_m, "type_of_meal_plan", "booking_status", figsize=(6, 3))
    return


@app.cell
def _(data_m, stacked_barplot):
    stacked_barplot(data_m, "room_type_reserved", "booking_status", figsize=(8, 4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `lead_time` distribution by cancellation status
    """)
    return


@app.cell
def _(data_m, distribution_plot_wrt_target):
    distribution_plot_wrt_target(data_m, "lead_time", "booking_status", figsize=(12, 6))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `avg_price_per_room` distribution by cancellation status
    """)
    return


@app.cell
def _(data_m, distribution_plot_wrt_target):
    distribution_plot_wrt_target(data_m, "avg_price_per_room", "booking_status", figsize=(12, 6))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Bivariate observations

    - **Busiest months**: bookings peak in October (5 317), September (4 611), and August (3 813). January (1 014) is the quietest; clear seasonality.
    - **Market segment distribution**: Online 64%, Offline 29%, Corporate 5.6%, Complementary 1%, Aviation <0.5%.
    - **Price by segment** (mean): Online 112 EUR, Aviation 101 EUR, Offline 92 EUR, Corporate 83 EUR, Complementary 3 EUR (promotional bookings explain the near-zero rate).
    - **Overall cancellation rate**: 32.8%. Moderate imbalance, stratify the split.
    - **Repeated guests**: cancel at 1.7% vs 33.6% for first-time guests, an order-of-magnitude retention signal.
    - **Special requests**: cancellation rate drops monotonically with the number of requests: 0 → 43%, 1 → 24%, 2 → 15%, 3+ → 0%. The strongest single categorical predictor.
    - **Market segment**: Online cancels at 36.5% (highest); Offline 29.9%; Corporate 10.9%; Complementary 0%.
    - **Lead time**: cancelled bookings have a mean lead time of 139 days vs 59 days for non-cancelled (median 122 vs 39). The strongest *numeric* predictor visible in EDA.
    - **Price**: cancelled bookings average 110.6 EUR vs 99.9 EUR for non-cancelled. Modest gap, partly driven by Online concentration.
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
    ## Data Preprocessing
    *Outlier inspection, feature engineering, scaling, encoding, split*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Outlier inspection

    Don't auto-cap. Reason per feature:

    - `avg_price_per_room` zeros - investigate (likely complementary segment)
    cap at upper IQR whisker only after confirming non-zero outliers are erroneous, not real luxury suites.
    - `lead_time` extremes - long-lead bookings are real usually keep.
    - `no_of_previous_cancellations` / `_not_canceled` - heavy-tailed but legitimate behavior.
    """)
    return


@app.cell
def _(data_m, np, plt):
    # Numeric-feature boxplot grid (excluding target)
    numeric_cols = data_m.select_dtypes(include=np.number).columns.tolist()
    numeric_cols.remove("booking_status")

    n = len(numeric_cols)
    cols_per_row = 4
    rows = (n + cols_per_row - 1) // cols_per_row

    plt.figure(figsize=(cols_per_row * 4, rows * 3))
    for i, col in enumerate(numeric_cols, start=1):
        plt.subplot(rows, cols_per_row, i)
        plt.boxplot(data_m[col], whis=1.5)
        plt.title(col, fontsize=10)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### `avg_price_per_room`: investigate zeros
    """)
    return


@app.cell
def _(data_m):
    # How many zero-priced rooms, and which segments?
    zero_price = data_m[data_m["avg_price_per_room"] == 0]
    print(f"Zero-priced bookings: {len(zero_price):,} ({len(zero_price) / len(data_m):.2%})")
    zero_price["market_segment_type"].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Zero prices concentrate in Complementary (promotional) and Online segments, they are legitimate domain values, not data-entry errors. Keeping them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Treat upper outliers in `avg_price_per_room`

    Cap values above the upper IQR whisker (`Q3 + 1.5·IQR`) - extreme high prices are likely data anomalies in this dataset rather than legitimate luxury suites (room types are encoded and rare types are well-represented at moderate prices).
    """)
    return


@app.cell
def _(data_m):
    # Compute upper whisker for avg_price_per_room
    Q1 = data_m["avg_price_per_room"].quantile(0.25)
    Q3 = data_m["avg_price_per_room"].quantile(0.75)
    IQR = Q3 - Q1
    upper_whisker = Q3 + 1.5 * IQR
    print(f"Q1={Q1:.2f}  Q3={Q3:.2f}  IQR={IQR:.2f}  Upper whisker={upper_whisker:.2f}")
    return (upper_whisker,)


@app.cell
def _(data_m, upper_whisker):
    # Cap upper outliers at the whisker value
    above = (data_m["avg_price_per_room"] > upper_whisker).sum()
    print(f"Capping {above:,} values above {upper_whisker:.2f}")
    data_m.loc[data_m["avg_price_per_room"] > upper_whisker, "avg_price_per_room"] = upper_whisker
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Other numeric features

    - `lead_time`: long lead times are real (annual planners, conference blocks). Keep.
    - `no_of_previous_cancellations` and `no_of_previous_bookings_not_canceled`: heavy-tailed but real customer history. Keep.
    - `no_of_children`: extreme values (e.g., 9, 10) likely data-entry typos.
    """)
    return


@app.cell
def _(data_m):
    # Replace extreme child counts with 3
    data_m["no_of_children"] = data_m["no_of_children"].replace([9, 10], 3)
    data_m["no_of_children"].value_counts().sort_index()
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
    ### Train-test split

    Stratified 80/20 (preserves cancellation rate in both folds)
    """)
    return


@app.cell
def _(data_m, train_test_split):
    X = data_m.drop(columns=["booking_status"])
    y = data_m["booking_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=47
    )
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Train class %:", y_train.value_counts(normalize=True).to_dict())
    print("Test  class %:", y_test.value_counts(normalize=True).to_dict())
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Encoding categoricals + scaling numerics

    Pipeline pattern: fit on train only, transform both. `StandardScaler` for numeric (KNN and SVM are scale-sensitive)
    one-hot with `drop_first=True` for categoricals.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    StandardScaler,
    X_test,
    X_train,
    np,
    pd,
):

    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    # Fit on train only (no leakage), transform both
    X_train_p_arr = preprocessor.fit_transform(X_train)
    X_test_p_arr = preprocessor.transform(X_test)

    # Recover feature names for downstream interpretability
    feature_names = (
        num_cols
        + preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    )

    X_train_p = pd.DataFrame(X_train_p_arr, columns=feature_names, index=X_train.index)
    X_test_p = pd.DataFrame(X_test_p_arr, columns=feature_names, index=X_test.index)

    print("Preprocessed shapes. train:", X_train_p.shape, "test:", X_test_p.shape)
    print(f"Numeric: {len(num_cols)}, Categorical (post-OHE): {len(feature_names) - len(num_cols)}")
    return X_test_p, X_train_p, feature_names


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ***
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Building
    *Baseline classifiers: KNN (k=3), Naive Bayes, SVM (linear)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### KNN - k=3 baseline
    """)
    return


@app.cell
def _(KNeighborsClassifier, X_train_p, y_train):
    knn_0 = KNeighborsClassifier(n_neighbors=3)
    knn_0.fit(X_train_p, y_train)
    return (knn_0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Train confusion matrix and metrics**
    """)
    return


@app.cell
def _(X_train_p, knn_0, plot_confusion_matrix, y_train):
    plot_confusion_matrix(knn_0, X_train_p, y_train)
    return


@app.cell
def _(X_train_p, classification_report_df, knn_0, y_train):
    knn_0_train = classification_report_df(knn_0, X_train_p, y_train)
    knn_0_train
    return (knn_0_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Test confusion matrix and metrics**
    """)
    return


@app.cell
def _(X_test_p, knn_0, plot_confusion_matrix, y_test):
    plot_confusion_matrix(knn_0, X_test_p, y_test)
    return


@app.cell
def _(X_test_p, classification_report_df, knn_0, y_test):
    knn_0_test = classification_report_df(knn_0, X_test_p, y_test)
    knn_0_test
    return (knn_0_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *KNN with `k=3` overfits on the training set (very high recall) but generalises noticeably less to the test set. We will sweep `k` to find a better trade-off.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Naive Bayes - GaussianNB
    """)
    return


@app.cell
def _(GaussianNB, X_train_p, y_train):
    nb = GaussianNB()
    nb.fit(X_train_p, y_train)
    return (nb,)


@app.cell
def _(X_train_p, nb, plot_confusion_matrix, y_train):
    plot_confusion_matrix(nb, X_train_p, y_train)
    return


@app.cell
def _(X_train_p, classification_report_df, nb, y_train):
    nb_train = classification_report_df(nb, X_train_p, y_train)
    nb_train
    return (nb_train,)


@app.cell
def _(X_test_p, nb, plot_confusion_matrix, y_test):
    plot_confusion_matrix(nb, X_test_p, y_test)
    return


@app.cell
def _(X_test_p, classification_report_df, nb, y_test):
    nb_test = classification_report_df(nb, X_test_p, y_test)
    nb_test
    return (nb_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Naive Bayes is fast and reasonable but its independence assumption is violated here (clear correlations between, e.g. `lead_time` and price/segment). It generalises poorly compared to KNN, a typical baseline.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SVM - linear kernel baseline
    """)
    return


@app.cell
def _(SVC, X_train_p, y_train):
    # Linear SVM baseline. probability=True is expensive (Platt CV at fit) but lets us
    # sweep thresholds later. random_state pinned for reproducibility.
    svm_linear = SVC(kernel="linear", probability=True, random_state=47)
    svm_linear.fit(X_train_p, y_train)
    return (svm_linear,)


@app.cell
def _(X_train_p, plot_confusion_matrix, svm_linear, y_train):
    plot_confusion_matrix(svm_linear, X_train_p, y_train)
    return


@app.cell
def _(X_train_p, classification_report_df, svm_linear, y_train):
    svm_linear_train = classification_report_df(svm_linear, X_train_p, y_train)
    svm_linear_train
    return (svm_linear_train,)


@app.cell
def _(X_test_p, plot_confusion_matrix, svm_linear, y_test):
    plot_confusion_matrix(svm_linear, X_test_p, y_test)
    return


@app.cell
def _(X_test_p, classification_report_df, svm_linear, y_test):
    svm_linear_test = classification_report_df(svm_linear, X_test_p, y_test)
    svm_linear_test
    return (svm_linear_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Linear SVM is a sensible starting point but a linear hyperplane likely under-models lead-time / segment / price interactions. We'll sweep kernels (poly d=2, d=3, RBF) with `C` and `gamma` next.*
    """)
    return


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
def _(
    KNeighborsClassifier,
    X_test_p,
    X_train_p,
    f1_score,
    plt,
    recall_score,
    y_test,
    y_train,
):

    neighbors = [k for k in range(3, 20) if k % 2 != 0]
    recall_train, recall_test, f1_train_list, f1_test_list = [], [], [], []
    for k in neighbors:
        knn_k = KNeighborsClassifier(n_neighbors=k)
        knn_k.fit(X_train_p, y_train)
        recall_train.append(recall_score(y_train, knn_k.predict(X_train_p)))
        recall_test.append(recall_score(y_test, knn_k.predict(X_test_p)))
        f1_train_list.append(f1_score(y_train, knn_k.predict(X_train_p)))
        f1_test_list.append(f1_score(y_test, knn_k.predict(X_test_p)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(neighbors, recall_train, marker="o", label="Train")
    ax1.plot(neighbors, recall_test, marker="o", label="Test")
    ax1.set(xlabel="k", ylabel="Recall (cancelled class)", title="KNN: recall vs k")
    ax1.set_xticks(neighbors)
    ax1.legend()
    ax1.grid(True)

    ax2.plot(neighbors, f1_train_list, marker="o", label="Train")
    ax2.plot(neighbors, f1_test_list, marker="o", label="Test")
    ax2.set(xlabel="k", ylabel="F1 (cancelled class)", title="KNN: F1 vs k")
    ax2.set_xticks(neighbors)
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Test recall is highest at `k=3` (~0.76) and decreases monotonically. F1 is roughly flat across `k ∈ {3, 5, 9}` (~0.77). The trade-off is generalisation: `k=3` overfits (train-test gap ~0.09), while `k≥9` generalises tightly (gap ≤0.03). We keep `k=9` as the tuned comparator; the final-model rationale below revisits `k=3` vs `k=9`.*
    """)
    return


@app.cell
def _(KNeighborsClassifier, X_train_p, y_train):
    # k=3 has highest test recall but a noticeable train-test gap (~0.09).
    # k=9 ties on test F1 (0.774) with a much smaller gap (~0.03), so we keep
    # it as the "tuned" comparator. Final-model rationale below covers k=3 vs k=9.
    K_BEST = 9
    knn_best = KNeighborsClassifier(n_neighbors=K_BEST)
    knn_best.fit(X_train_p, y_train)
    return K_BEST, knn_best


@app.cell
def _(
    X_train_p,
    classification_report_df,
    knn_best,
    plot_confusion_matrix,
    y_train,
):
    plot_confusion_matrix(knn_best, X_train_p, y_train)
    knn_best_train = classification_report_df(knn_best, X_train_p, y_train)
    knn_best_train
    return (knn_best_train,)


@app.cell
def _(
    X_test_p,
    classification_report_df,
    knn_best,
    plot_confusion_matrix,
    y_test,
):
    plot_confusion_matrix(knn_best, X_test_p, y_test)
    knn_best_test = classification_report_df(knn_best, X_test_p, y_test)
    knn_best_test
    return (knn_best_test,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SVM tuning: kernel and parameters

    SVM combinations across `kernel ∈ {linear, poly d=2, poly d=3, rbf}`, `gamma ∈ {default, 0.016}`, `C ∈ {default, 0.1}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Each SVM variant below trains on the full training set with `probability=True`.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **SVM 1 - Polynomial kernel, degree = 2**
    """)
    return


@app.cell
def _(SVC, X_train_p, y_train):
    svm_poly2 = SVC(kernel="poly", degree=2, probability=True, random_state=47)
    svm_poly2.fit(X_train_p, y_train)
    return (svm_poly2,)


@app.cell
def _(
    X_test_p,
    X_train_p,
    classification_report_df,
    svm_poly2,
    y_test,
    y_train,
):
    svm_poly2_train = classification_report_df(svm_poly2, X_train_p, y_train)
    svm_poly2_test = classification_report_df(svm_poly2, X_test_p, y_test)
    print("Train:")

    print(svm_poly2_train)
    print("\nTest:")

    print(svm_poly2_test)
    return svm_poly2_test, svm_poly2_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **SVM 2 - Polynomial kernel, degree = 2, gamma = 0.016**
    """)
    return


@app.cell
def _(SVC, X_train_p, y_train):
    svm_poly2_g = SVC(kernel="poly", degree=2, gamma=0.016, probability=True, random_state=47)
    svm_poly2_g.fit(X_train_p, y_train)
    return (svm_poly2_g,)


@app.cell
def _(
    X_test_p,
    X_train_p,
    classification_report_df,
    svm_poly2_g,
    y_test,
    y_train,
):
    svm_poly2_g_train = classification_report_df(svm_poly2_g, X_train_p, y_train)
    svm_poly2_g_test = classification_report_df(svm_poly2_g, X_test_p, y_test)
    print("Train:")

    print(svm_poly2_g_train)
    print("\nTest:")

    print(svm_poly2_g_test)
    return svm_poly2_g_test, svm_poly2_g_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **SVM 3 - Polynomial kernel, degree = 3**
    """)
    return


@app.cell
def _(SVC, X_train_p, y_train):
    svm_poly3 = SVC(kernel="poly", degree=3, probability=True, random_state=47)
    svm_poly3.fit(X_train_p, y_train)
    return (svm_poly3,)


@app.cell
def _(
    X_test_p,
    X_train_p,
    classification_report_df,
    svm_poly3,
    y_test,
    y_train,
):
    svm_poly3_train = classification_report_df(svm_poly3, X_train_p, y_train)
    svm_poly3_test = classification_report_df(svm_poly3, X_test_p, y_test)
    print("Train:")

    print(svm_poly3_train)
    print("\nTest:")

    print(svm_poly3_test)
    return svm_poly3_test, svm_poly3_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **SVM 4 - RBF kernel (default gamma)**
    """)
    return


@app.cell
def _(SVC, X_train_p, y_train):
    svm_rbf = SVC(kernel="rbf", probability=True, random_state=47)
    svm_rbf.fit(X_train_p, y_train)
    return (svm_rbf,)


@app.cell
def _(X_test_p, X_train_p, classification_report_df, svm_rbf, y_test, y_train):
    svm_rbf_train = classification_report_df(svm_rbf, X_train_p, y_train)
    svm_rbf_test = classification_report_df(svm_rbf, X_test_p, y_test)
    print("Train:")

    print(svm_rbf_train)
    print("\nTest:")

    print(svm_rbf_test)
    return svm_rbf_test, svm_rbf_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **SVM 5 - RBF kernel, gamma = 0.016**
    """)
    return


@app.cell
def _(SVC, X_train_p, y_train):
    svm_rbf_g = SVC(kernel="rbf", gamma=0.016, probability=True, random_state=47)
    svm_rbf_g.fit(X_train_p, y_train)
    return (svm_rbf_g,)


@app.cell
def _(
    X_test_p,
    X_train_p,
    classification_report_df,
    svm_rbf_g,
    y_test,
    y_train,
):
    svm_rbf_g_train = classification_report_df(svm_rbf_g, X_train_p, y_train)
    svm_rbf_g_test = classification_report_df(svm_rbf_g, X_test_p, y_test)
    print("Train:")

    print(svm_rbf_g_train)
    print("\nTest:")

    print(svm_rbf_g_test)
    return svm_rbf_g_test, svm_rbf_g_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **SVM 6 - RBF kernel, gamma = 0.016, C = 0.1**
    """)
    return


@app.cell
def _(SVC, X_train_p, y_train):
    svm_rbf_g_c = SVC(kernel="rbf", gamma=0.016, C=0.1, probability=True, random_state=47)
    svm_rbf_g_c.fit(X_train_p, y_train)
    return (svm_rbf_g_c,)


@app.cell
def _(
    X_test_p,
    X_train_p,
    classification_report_df,
    svm_rbf_g_c,
    y_test,
    y_train,
):
    svm_rbf_g_c_train = classification_report_df(svm_rbf_g_c, X_train_p, y_train)
    svm_rbf_g_c_test = classification_report_df(svm_rbf_g_c, X_test_p, y_test)
    print("Train:")

    print(svm_rbf_g_c_train)
    print("\nTest:")

    print(svm_rbf_g_c_test)
    return svm_rbf_g_c_test, svm_rbf_g_c_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Performance Comparison
    *Side-by-side train and test metrics for all variants*
    """)
    return


@app.cell
def _(
    K_BEST,
    knn_0_test,
    knn_0_train,
    knn_best_test,
    knn_best_train,
    nb_test,
    nb_train,
    pd,
    svm_linear_test,
    svm_linear_train,
    svm_poly2_g_test,
    svm_poly2_g_train,
    svm_poly2_test,
    svm_poly2_train,
    svm_poly3_test,
    svm_poly3_train,
    svm_rbf_g_c_test,
    svm_rbf_g_c_train,
    svm_rbf_g_test,
    svm_rbf_g_train,
    svm_rbf_test,
    svm_rbf_train,
):
    cols = [
        "KNN k=3",
        f"KNN k={K_BEST}",
        "Naive Bayes",
        "SVM linear",
        "SVM poly d=2",
        "SVM poly d=2, γ=0.016",
        "SVM poly d=3",
        "SVM RBF",
        "SVM RBF, γ=0.016",
        "SVM RBF, γ=0.016, C=0.1",
    ]

    train_table = pd.concat(
        [
            knn_0_train.T, knn_best_train.T, nb_train.T, svm_linear_train.T,
            svm_poly2_train.T, svm_poly2_g_train.T, svm_poly3_train.T,
            svm_rbf_train.T, svm_rbf_g_train.T, svm_rbf_g_c_train.T,
        ],
        axis=1,
    )
    train_table.columns = cols

    test_table = pd.concat(
        [
            knn_0_test.T, knn_best_test.T, nb_test.T, svm_linear_test.T,
            svm_poly2_test.T, svm_poly2_g_test.T, svm_poly3_test.T,
            svm_rbf_test.T, svm_rbf_g_test.T, svm_rbf_g_c_test.T,
        ],
        axis=1,
    )
    test_table.columns = cols
    return test_table, train_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Train performance**
    """)
    return


@app.cell
def _(train_table):
    train_table.T.style.format("{:.3f}").background_gradient(cmap="Blues")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Test performance**
    """)
    return


@app.cell
def _(test_table):
    test_table.T.style.format("{:.3f}").background_gradient(cmap="Greens")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Final model selection

    Pick the model with the **highest test recall** under an acceptable train-test gap. Tie-breakers: training cost (KNN has none, SVM-RBF has high cost on 36k rows), simplicity, deployment fit.
    """)
    return


@app.cell
def _(knn_0):
    # KNN k=3 wins on this dataset:
    #   - Highest non-degenerate test recall (~0.76)
    #   - Highest test F1 (~0.77), best balance of recall and precision
    #   - SVM RBF (next best non-NB) lands at recall ~0.71 / F1 ~0.76
    #
    # Naive Bayes has the highest raw test recall (~0.96), but precision is ~0.35, i.e.
    # it predicts ~all bookings as cancellations. Useful only if false alarms are free
    # (pure retention nudge); not appropriate for overbooking or refund decisions.
    best_estimator = knn_0
    return (best_estimator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Why not Naive Bayes?** NB precision is ~0.35: 2 of every 3 flagged bookings are false alarms. Acceptable for a retention nudge, not for overbooking or refund-policy decisions.

    **Why k=3 over k=9?** k=3 wins on test recall (0.760 vs 0.746) and ties on F1 (0.774). The k=3 train-test gap is wider (0.09 vs 0.03), so k=9 is a defensible production-stability alternative.

    **Why not SVM RBF?** Test recall (0.71) and F1 (0.76) are both lower than KNN k=3 at comparable cost.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Threshold tuning

    The default 0.5 threshold is rarely optimal under cost-asymmetric framing. Sweep the threshold on validation, plot precision-recall curve, pick the operating point matching the business constraint.
    """)
    return


@app.cell
def _(X_test_p, best_estimator, plt, precision_recall_curve, y_test):
    proba = best_estimator.predict_proba(X_test_p)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision and recall vs threshold (test set)")
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    KNN with `k=3` and uniform weights produces only **four distinct probabilities**: 0, 1/3, 2/3, 1.0. The precision-recall curve has the same number of operating points:

    | Threshold | Precision | Recall |
    |---|---|---|
    | 0.000 | 0.328 | 1.000 |
    | **0.333** | **0.626** | **0.897** |
    | 0.667 (default) | 0.788 | 0.760 |
    | 1.000 | 0.913 | 0.577 |

    **Pick `THRESHOLD = 0.333`** to hit ~90% recall on the cancelled class at ~63% precision. A strong operating point for the cost-asymmetric framing: catch nearly all cancellations, accept lower precision as cheap retention outreach.

    Higher-resolution thresholding would require either `weights='distance'` (smooth probabilities) or switching to SVM with `probability=True`. Both are possible follow-ups.
    """)
    return


@app.cell
def _(
    X_test_p,
    best_estimator,
    classification_report_predict_proba,
    plot_confusion_matrix_proba,
    y_test,
):
    THRESHOLD = 0.333
    plot_confusion_matrix_proba(best_estimator, X_test_p, y_test, threshold=THRESHOLD)
    tuned = classification_report_predict_proba(best_estimator, X_test_p, y_test, threshold=THRESHOLD)
    print(f"Threshold = {THRESHOLD}")
    tuned
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Permutation feature importance

    Model-agnostic feature importance under the chosen metric (what the *fitted* model relies on).
    """)
    return


@app.cell
def _(
    X_test_p,
    best_estimator,
    feature_names,
    pd,
    permutation_importance,
    plt,
    y_test,
):
    perm = permutation_importance(
        best_estimator, X_test_p, y_test,
        scoring="recall", n_repeats=5, random_state=47, n_jobs=-1,
    )

    importance = (
        pd.DataFrame(
            {"feature": feature_names, "importance": perm.importances_mean, "std": perm.importances_std}
        )
        .sort_values("importance", ascending=True)
    )

    plt.figure(figsize=(8, max(4, 0.3 * len(importance))))
    plt.barh(importance["feature"], importance["importance"], xerr=importance["std"], color="steelblue")
    plt.xlabel("Permutation importance (Δ recall)")
    plt.title("Permutation feature importance (scored on recall)")
    plt.tight_layout()
    plt.show()
    return (importance,)


@app.cell
def _(importance):
    importance.tail(15).iloc[::-1].reset_index(drop=True)
    return


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

    1. **Lead time is the dominant predictor.** Cancelled bookings average 139 days lead time vs 59 days for non-cancelled. Top permutation-importance feature on KNN k=3.
    2. **Special requests strongly suppress cancellations.** Rate falls monotonically with request count: 43% (0) → 24% (1) → 15% (2) → 0% (3+).
    3. **Repeated guests are loyal.** 1.7% cancel vs 33.6% for first-time guests, but only 2.6% of bookings are repeat-guest.
    4. **Cancellation risk concentrates in the Online segment.** Online 36.5%, Offline 29.9%, Corporate 10.9%, Complementary 0%.
    5. **Price has a modest direct effect.** Cancelled bookings average 110.6 EUR vs 99.9 EUR. Most of the signal is mediated by segment.
    6. **Volume concentrates Aug–Oct.** Peak months are 3–5× quieter ones, so absolute cancellation cost concentrates there even at a constant rate.
    7. **Final model.** KNN k=3, test recall 0.76 / F1 0.77. Threshold 0.333 lifts recall to 0.90 at 0.63 precision.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Business Recommendations

    1. **Tiered cancellation policy by lead time.** Apply non-refundable deposits or stricter terms to bookings with `lead_time > 120 days`. Highest-leverage lever surfaced by the model.
    2. **Reward special requests at booking.** A guided "personalise your stay" flow that lifts the request count converts soft holds into committed bookings.
    3. **Channel-specific refund terms.** Apply stricter refund terms on the Online segment; keep Corporate and Complementary flexible.
    4. **Loyalty programme for repeat guests.** Repeat guests cancel at 1.7%. Retention investment (priority upgrades, returning-guest perks) compounds.
    5. **Risk-tiered overbooking on peak dates.** Use the model's cancellation probability on Aug–Oct dates with Online-segment volume. Threshold 0.333 catches ~9 of 10 cancellations at ~63% precision. Cap overbooking so displacement cost stays below expected recovered revenue.
    6. **Treat the threshold as a policy lever.** Document the operating point (0.333) alongside the model and revisit quarterly. Higher precision suits revenue-management decisions; higher recall suits retention nudges.
    7. **Naive Bayes as a retention-nudge layer.** 96% recall at 35% precision. Defensible for cheap email/SMS first-touch on top of KNN overbooking decisions.

    ***
    """)
    return


if __name__ == "__main__":
    app.run()
