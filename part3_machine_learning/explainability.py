import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# -------------------------------------------------
# Load Part 2 feature-engineered dataset
# -------------------------------------------------
def load_feature_data(csv_path):
    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=["date_time"],
        )

        logger.info(
            "Feature-engineered dataset loaded: "
            "%d rows, %d columns.",
            df.shape[0],
            df.shape[1],
        )

        return df

    except (FileNotFoundError, pd.errors.ParserError):
        logger.error(
            "Unable to load feature-engineered dataset: %s",
            csv_path,
            exc_info=True,
        )
        return None


# -------------------------------------------------
# Prepare same feature set used for regression
# -------------------------------------------------
def prepare_model_features(df):
    df = df.copy()

    df["holiday_flag"] = (
        df["holiday"]
        .str.lower()
        .ne("none")
        .astype(int)
    )

    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    feature_columns = [
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "weekend",
        "holiday_flag",
        "temp_scaled",
        "rain_1h",
        "snow_1h",
        "clouds_all",
    ]

    weather_columns = [
        col for col in df.columns
        if col.startswith("weather_")
        and col not in [
            "weather_main",
            "weather_description",
        ]
    ]

    feature_columns.extend(weather_columns)

    logger.info(
        "Explainability feature set prepared "
        "with %d features.",
        len(feature_columns),
    )

    return df, feature_columns


# -------------------------------------------------
# Train comparable Random Forest regression model
# -------------------------------------------------
def train_random_forest(df, feature_columns):
    X = df[feature_columns]
    y = df["traffic_volume"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    logger.info(
        "Random Forest Regressor trained "
        "for SHAP explainability."
    )

    return model, X_train, X_test


# -------------------------------------------------
# Apply SHAP explainability
# -------------------------------------------------
def apply_shap(
    model,
    X_train,
    X_test,
):
    figures_dir = Path(
        "part3_machine_learning/figures"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Use a manageable sample for SHAP
    background = X_train.sample(
        n=min(100, len(X_train)),
        random_state=RANDOM_STATE,
    )

    explain_data = X_test.sample(
        n=min(200, len(X_test)),
        random_state=RANDOM_STATE,
    )

    explainer = shap.Explainer(
        model.predict,
        background,
        algorithm="permutation",
        feature_names=X_train.columns.tolist(),
    )

    shap_values = explainer(
        explain_data,
        max_evals=2 * len(X_train.columns) + 1,
    )

    logger.info(
        "SHAP values calculated for %d test records.",
        len(explain_data),
    )

    # ---------------------------------------------
    # SHAP summary plot
    # ---------------------------------------------
    shap.summary_plot(
        shap_values.values,
        explain_data,
        show=False,
    )

    summary_path = (
        figures_dir
        / "task3_shap_summary.png"
    )

    plt.tight_layout()
    plt.savefig(
        summary_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    logger.info(
        "SHAP summary plot saved: %s",
        summary_path,
    )

    # ---------------------------------------------
    # SHAP feature importance bar plot
    # ---------------------------------------------
    shap.summary_plot(
        shap_values.values,
        explain_data,
        plot_type="bar",
        show=False,
    )

    importance_path = (
        figures_dir
        / "task3_shap_feature_importance.png"
    )

    plt.tight_layout()
    plt.savefig(
        importance_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    logger.info(
        "SHAP feature-importance plot saved: %s",
        importance_path,
    )


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    csv_path = Path(
        "data/processed/traffic_features_part2.csv"
    )

    df = load_feature_data(csv_path)

    if df is None:
        sys.exit(1)

    df, feature_columns = prepare_model_features(
        df
    )

    model, X_train, X_test = (
        train_random_forest(
            df,
            feature_columns,
        )
    )

    apply_shap(
        model,
        X_train,
        X_test,
    )


# -------------------------------------------------
# Logging configuration
# -------------------------------------------------
if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - "
        "%(funcName)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        "part3_machine_learning/part3.log"
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    main()