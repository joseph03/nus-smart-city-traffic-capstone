import logging
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# -------------------------------------------------
# Weather categories used for proxy accident risk
# -------------------------------------------------
SEVERE_WEATHER = [
    "Thunderstorm",
    "Snow",
    "Squall",
]

LOW_VISIBILITY_WEATHER = [
    "Fog",
    "Mist",
    "Smoke",
    "Haze",
]


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
# Create Part 3 classification target
# -------------------------------------------------
def create_high_risk_label(df):
    df = df.copy()

    q1, q2, q3 = df["traffic_volume"].quantile(
        [0.25, 0.50, 0.75]
    ).values

    def congestion_bucket(value):
        if value <= q1:
            return "Low"
        elif value <= q2:
            return "Medium"
        elif value <= q3:
            return "High"
        else:
            return "Severe"

    df["congestion_category_part3"] = (
        df["traffic_volume"].apply(
            congestion_bucket
        )
    )

    df["is_low_visibility"] = (
        df["weather_main"]
        .isin(LOW_VISIBILITY_WEATHER)
        .astype(int)
    )

    high_congestion = (
        df["congestion_category_part3"]
        .isin(["High", "Severe"])
    )

    risky_weather = (
        df["weather_main"].isin(
            SEVERE_WEATHER
        )
        | (df["is_low_visibility"] == 1)
    )

    df["high_risk"] = (
        high_congestion & risky_weather
    ).astype(int)

    logger.info(
        "Proxy high_risk label created."
    )

    return df


# -------------------------------------------------
# Prepare same 21-feature set used in Task 1
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
        "MLflow feature set prepared "
        "with %d features.",
        len(feature_columns),
    )

    return df, feature_columns


# -------------------------------------------------
# Track classification models
# -------------------------------------------------
def track_classification_models(
    df,
    feature_columns,
):
    X = df[feature_columns]
    y = df["high_risk"]

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ---------------------------------------------
    # Logistic Regression
    # ---------------------------------------------
    logistic_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    with mlflow.start_run(
        run_name="Logistic Regression Classification"
    ):
        logistic_model.fit(
            X_train,
            y_train,
        )

        predictions = logistic_model.predict(
            X_test
        )

        probabilities = (
            logistic_model.predict_proba(
                X_test
            )[:, 1]
        )

        metrics = {
            "accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(
                y_test,
                probabilities,
            ),
        }

        mlflow.log_param(
            "model_type",
            "LogisticRegression",
        )

        mlflow.log_param(
            "max_iter",
            1000,
        )

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            logistic_model,
            name="model",
        )

        logger.info(
            "MLflow run completed: "
            "Logistic Regression Classification."
        )

    # ---------------------------------------------
    # Random Forest Classifier
    # ---------------------------------------------
    random_forest_model = (
        RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    with mlflow.start_run(
        run_name="Random Forest Classification"
    ):
        random_forest_model.fit(
            X_train,
            y_train,
        )

        predictions = (
            random_forest_model.predict(
                X_test
            )
        )

        probabilities = (
            random_forest_model.predict_proba(
                X_test
            )[:, 1]
        )

        metrics = {
            "accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(
                y_test,
                probabilities,
            ),
        }

        mlflow.log_param(
            "model_type",
            "RandomForestClassifier",
        )

        mlflow.log_param(
            "n_estimators",
            200,
        )

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            random_forest_model,
            name="model",
        )

        logger.info(
            "MLflow run completed: "
            "Random Forest Classification."
        )


# -------------------------------------------------
# Track regression models
# -------------------------------------------------
def track_regression_models(
    df,
    feature_columns,
):
    X = df[feature_columns]
    y = df["traffic_volume"]

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    # ---------------------------------------------
    # Linear Regression
    # ---------------------------------------------
    linear_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                LinearRegression(),
            ),
        ]
    )

    with mlflow.start_run(
        run_name="Linear Regression"
    ):
        linear_model.fit(
            X_train,
            y_train,
        )

        predictions = linear_model.predict(
            X_test
        )

        metrics = {
            "mae": mean_absolute_error(
                y_test,
                predictions,
            ),
            "r2": r2_score(
                y_test,
                predictions,
            ),
        }

        mlflow.log_param(
            "model_type",
            "LinearRegression",
        )

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            linear_model,
            name="model",
        )

        logger.info(
            "MLflow run completed: "
            "Linear Regression."
        )

    # ---------------------------------------------
    # Random Forest Regressor
    # ---------------------------------------------
    random_forest_model = (
        RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    with mlflow.start_run(
        run_name="Random Forest Regression"
    ):
        random_forest_model.fit(
            X_train,
            y_train,
        )

        predictions = (
            random_forest_model.predict(
                X_test
            )
        )

        metrics = {
            "mae": mean_absolute_error(
                y_test,
                predictions,
            ),
            "r2": r2_score(
                y_test,
                predictions,
            ),
        }

        mlflow.log_param(
            "model_type",
            "RandomForestRegressor",
        )

        mlflow.log_param(
            "n_estimators",
            200,
        )

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            random_forest_model,
            name="model",
        )

        logger.info(
            "MLflow run completed: "
            "Random Forest Regression."
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

    df = create_high_risk_label(df)

    df, feature_columns = (
        prepare_model_features(df)
    )

    mlflow_db = Path(
        "part3_machine_learning/mlflow.db"
    )

    mlflow.set_tracking_uri(
        f"sqlite:///{mlflow_db.resolve()}"
    )

    mlflow.set_experiment(
        "smart_city_traffic_models"
    )

    logger.info(
        "MLflow experiment: "
        "smart_city_traffic_models"
    )

    track_classification_models(
        df,
        feature_columns,
    )

    track_regression_models(
        df,
        feature_columns,
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