"""src/datafun/app.py - Project script.

Author: Denise Case
Date: 2026-09

HOW TO RUN THIS FILE:

From the VS Code menu (with only this project open in VS Code),
click "Terminal" / New Terminal to open an integrated Terminal
in the root project folder.

Paste the following command and press ENTER or RETURN:

uv run python -m datafun.app

DOMAIN:

A dataset of penguins.
See docs/data-card.md for more information about the dataset.

EXPLORE:

Earlier analysis showed relationships among
numeric penguin measurements.

In this project, we use TWO numeric features
to predict one numeric target
with a multiple linear regression model.

A standard predictive modeling process is:

1. OBSERVE the data and prior findings.
2. DECLARE the target and features.
3. PREPARE the modeling data.
4. SPLIT into training and test data.
5. BASELINE with a simple reference model.
6. TRAIN a LinearRegression model.
7. PREDICT on X_test.
8. EVALUATE baseline vs model on y_test.
9. VISUALIZE predictions and residuals.
10. ASSESS the results.

DESIGN:

Use this file to declare the data-specific choices
and the reasoning behind them,
then orchestrate the work.

Scikit-learn provides the machine learning tools.

The target, features, split, baseline,
and model choices stay here because they are
analytical decisions specific to this project.
"""

# === DECLARE IMPORTS (BRING IN FREE CODE) ===

import logging
from pathlib import Path
from typing import Final

from datafun_toolkit.logger import get_logger, log_header, log_path
import matplotlib.pyplot as plt
from ml_vizkit import save_chart
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


# === CONFIGURE LOGGER ONCE FOR THE APPLICATION ===

LOG: logging.Logger = get_logger("P06", level="DEBUG")


# === DECLARE GLOBAL CONSTANTS ===

DATA_FILE_PATH: Final[Path] = Path("data") / "raw" / "penguins.csv"

CHART_DIR: Final[Path] = Path("docs") / "images"

PREDICTION_CHART_PATH: Final[Path] = (
    CHART_DIR / "multiple-regression-predictions.png"
)

RESIDUAL_CHART_PATH: Final[Path] = (
    CHART_DIR / "multiple-regression-residuals.png"
)


# === DETERMINE WHAT ONE ROW REPRESENTS ===

GRAIN: Final[str] = "one penguin"


# === DECLARE THE TARGET ===

TARGET_COLUMN: Final[str] = "body_mass_g"


# === DECLARE THE FEATURES ===

# CUSTOM:
# Use two numeric features to predict body mass.

FEATURE_COLUMNS: Final[list[str]] = [
    "bill_length_mm",
    "flipper_length_mm",
]


# === DOCUMENT WHY THE FEATURES MIGHT HELP ===

FEATURE_DECISION: Final[str] = r"""
I want to predict body mass.

I selected bill length and flipper length as the features.

A larger penguin may have both a longer bill and
a longer flipper, so using both measurements may
provide more information for predicting body mass.

I do not know yet whether using two features will
predict body mass better than using a simple baseline.
The modeling process will provide evidence.
"""


# === DECLARE THE TRAIN / TEST SPLIT ===

TEST_FRACTION: Final[float] = 0.20

RANDOM_SEED: Final[int] = 42


# === DOCUMENT THE SPLIT DECISION ===

SPLIT_DECISION: Final[str] = r"""
I will use 80% of the modeling rows for training
and hold back 20% for testing.

I want most of the available data to be available
for learning the model, while still keeping a
separate set of observations that the model did
not see during training.

The test rows will be used later to evaluate how
the trained model performs on unseen observations.

I will use a random seed of 42.

The specific value 42 is not analytically important.
I use a fixed seed so the random split is reproducible.
"""


# === DECLARE THE BASELINE ===

BASELINE_STRATEGY: Final[str] = "mean"


# === DOCUMENT THE BASELINE DECISION ===

BASELINE_DECISION: Final[str] = r"""
Before evaluating the LinearRegression model,
I need a simple baseline for comparison.

The baseline will ignore the selected features
and predict the average body mass from the training
data for every test observation.

A useful predictive model should improve
on this simple reference prediction.
"""


# === DOCUMENT THE MODEL DECISION ===

MODEL_DECISION: Final[str] = r"""
I will use LinearRegression with two features.

The model will use bill length and flipper length
together to predict body mass.

Using two features allows me to test whether combining
two penguin measurements provides useful information
for predicting body mass.

The evaluation metrics and residual plot will help
assess whether the model is useful.
"""


# === DEFINE THE MAIN FUNCTION ===


def main() -> None:
    """Entry point when running this file as a Python script.

    This is where the instructions begin.

    Arguments:
        None.

    Returns:
        None.
    """

    log_header(LOG, "P06 - MULTIPLE LINEAR REGRESSION")

    LOG.info("===================================")
    LOG.info("START main()")
    LOG.info("===================================")

    LOG.info("-------------------------------")
    LOG.info("01. OBSERVE the data and prior findings.")
    LOG.info("-------------------------------")

    log_path(LOG, "data file", path=DATA_FILE_PATH)

    df: pd.DataFrame = pd.read_csv(DATA_FILE_PATH)

    LOG.info("Data loaded successfully.")
    LOG.info(f"Grain: {GRAIN}")
    LOG.info(f"Rows: {df.shape[0]}")
    LOG.info(f"Columns: {df.shape[1]}")
    LOG.info(f"Column names: {df.columns.tolist()}")

    LOG.info("-------------------------------")
    LOG.info("02. DECLARE the target and features.")
    LOG.info("-------------------------------")

    LOG.info(f"Target: {TARGET_COLUMN}")
    LOG.info(f"Features: {FEATURE_COLUMNS}")
    LOG.info(FEATURE_DECISION)

    LOG.info("-------------------------------")
    LOG.info("03. PREPARE the modeling data.")
    LOG.info("-------------------------------")

    # A regression model requires numeric values
    # for the selected features and target.
    #
    # The schema shows these measurement fields as VARCHAR,
    # so explicitly convert them to numeric values.

    modeling_columns: list[str] = [
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]

    df_model: pd.DataFrame = df.copy()

    for column in modeling_columns:
        df_model[column] = pd.to_numeric(
            df_model[column],
            errors="coerce",
        )

    # Keep only complete feature/target combinations.

    df_model = df_model.dropna(
        subset=modeling_columns
    ).copy()

    count_original: int = df.shape[0]
    count_model: int = df_model.shape[0]
    count_dropped: int = count_original - count_model

    LOG.info(f"Original rows: {count_original}")
    LOG.info(f"Modeling rows: {count_model}")
    LOG.info(f"Rows dropped: {count_dropped}")

    # X contains TWO feature columns.
    # y contains the target.

    X: pd.DataFrame = df_model[FEATURE_COLUMNS]

    y: pd.Series = df_model[TARGET_COLUMN]

    LOG.info(f"X shape: {X.shape}")
    LOG.info(f"y shape: {y.shape}")

    LOG.info("-------------------------------")
    LOG.info("04. SPLIT into training and test data.")
    LOG.info("-------------------------------")

    LOG.info(SPLIT_DECISION)

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
    )

    LOG.info(f"Training rows: {X_train.shape[0]}")
    LOG.info(f"Test rows: {X_test.shape[0]}")

    LOG.info("-------------------------------")
    LOG.info("05. BASELINE with a simple reference model.")
    LOG.info("-------------------------------")

    LOG.info(BASELINE_DECISION)

    baseline_model = DummyRegressor(
        strategy=BASELINE_STRATEGY,
    )

    baseline_model.fit(
        X_train,
        y_train,
    )

    baseline_predictions: np.ndarray = baseline_model.predict(
        X_test
    )

    baseline_rmse: float = float(
        root_mean_squared_error(
            y_test,
            baseline_predictions,
        )
    )

    baseline_r_squared: float = float(
        r2_score(
            y_test,
            baseline_predictions,
        )
    )

    LOG.info(
        f"Baseline strategy: {BASELINE_STRATEGY}"
    )
    LOG.info(
        f"Baseline RMSE: {baseline_rmse:.2f}"
    )
    LOG.info(
        f"Baseline R-squared: {baseline_r_squared:.3f}"
    )

    LOG.info("-------------------------------")
    LOG.info("06. TRAIN a multiple LinearRegression model.")
    LOG.info("-------------------------------")

    LOG.info(MODEL_DECISION)

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    coefficients: np.ndarray = model.coef_
    intercept: float = float(model.intercept_)

    LOG.info("The model learned these coefficients:")

    for feature, coefficient in zip(
        FEATURE_COLUMNS,
        coefficients,
    ):
        LOG.info(
            f"{feature}: {coefficient:.3f}"
        )

    LOG.info(
        f"Intercept: {intercept:.3f}"
    )

    LOG.info("-------------------------------")
    LOG.info("07. PREDICT on X_test.")
    LOG.info("-------------------------------")

    model_predictions: np.ndarray = model.predict(X_test)

    LOG.info(
        f"Predictions created: {len(model_predictions)}"
    )

    LOG.info("-------------------------------")
    LOG.info("08. EVALUATE baseline vs model on y_test.")
    LOG.info("-------------------------------")

    model_rmse: float = float(
        root_mean_squared_error(
            y_test,
            model_predictions,
        )
    )

    model_r_squared: float = float(
        r2_score(
            y_test,
            model_predictions,
        )
    )

    LOG.info("BASELINE RESULTS")
    LOG.info(
        f"RMSE:      {baseline_rmse:.2f}"
    )
    LOG.info(
        f"R-squared: {baseline_r_squared:.3f}"
    )

    LOG.info("MULTIPLE LINEAR REGRESSION RESULTS")
    LOG.info(
        f"RMSE:      {model_rmse:.2f}"
    )
    LOG.info(
        f"R-squared: {model_r_squared:.3f}"
    )

    LOG.info("-------------------------------")
    LOG.info("09. VISUALIZE predictions and residuals.")
    LOG.info("-------------------------------")

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # === ACTUAL VS. PREDICTED CHART ===

    _prediction_figure, prediction_ax = plt.subplots()

    y_test_values: np.ndarray = y_test.to_numpy()

    prediction_ax.scatter(
        y_test_values,
        model_predictions,
        label="Predicted",
    )

    # Reference line showing perfect predictions.

    prediction_min = min(
        y_test_values.min(),
        model_predictions.min(),
    )

    prediction_max = max(
        y_test_values.max(),
        model_predictions.max(),
    )

    prediction_ax.plot(
        [prediction_min, prediction_max],
        [prediction_min, prediction_max],
        linestyle="--",
        label="Perfect Prediction",
    )

    prediction_ax.set_title(
        "Actual vs. Predicted Penguin Body Mass"
    )

    prediction_ax.set_xlabel(
        "Actual Body Mass (g)"
    )

    prediction_ax.set_ylabel(
        "Predicted Body Mass (g)"
    )

    prediction_ax.legend()

    save_chart(
        prediction_ax,
        PREDICTION_CHART_PATH,
    )

    LOG.info(
        f"Chart saved successfully at {PREDICTION_CHART_PATH}."
    )

    # === RESIDUAL CHART ===

    residuals: np.ndarray = (
        y_test_values - model_predictions
    )

    _residual_figure, residual_ax = plt.subplots()

    residual_ax.scatter(
        model_predictions,
        residuals,
    )

    # Draw a horizontal reference line at zero.

    residual_ax.axhline(0)

    residual_ax.set_title(
        "Residuals for Multiple Feature Model"
    )

    residual_ax.set_xlabel(
        "Predicted Body Mass (g)"
    )

    residual_ax.set_ylabel(
        "Residual (Actual - Predicted Body Mass)"
    )

    save_chart(
        residual_ax,
        RESIDUAL_CHART_PATH,
    )

    LOG.info(
        f"Chart saved successfully at {RESIDUAL_CHART_PATH}."
    )

    # ============================================================
    # 10. ASSESS
    # ============================================================

    LOG.info("-------------------------------")
    LOG.info("10. ASSESS the results.")
    LOG.info("-------------------------------")

    LOG.info(
        r"""CUSTOM OBSERVATIONS:
    I used bill length and flipper length together
    to predict body mass.

    The baseline RMSE was recorded above.

    The multiple linear regression RMSE was recorded above.

    Compared with the baseline, the LinearRegression model
    should be evaluated based on whether it produced a
    lower RMSE and a stronger R-squared result.

    The actual vs. predicted chart shows how closely
    the predictions compare with the observed body mass.

    The residual plot shows how far the predictions
    were from the actual values.

    Based on these results, I can determine whether
    using two features provides useful information
    for predicting body mass.
    """
    )

    # ============================================================
    # DISPLAY
    # ============================================================

    LOG.info(
        "In a script, call plt.show() at the end to display all charts."
    )

    LOG.info(
        "Close all chart windows with the close button to continue."
    )

    plt.show()

    LOG.info("===================================")
    LOG.info("END main() - Executed successfully!")
    LOG.info("===================================")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()
