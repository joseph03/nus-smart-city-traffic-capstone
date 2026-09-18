import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Alert thresholds for the monitoring simulation
ERROR_INCREASE_THRESHOLD = 0.25
FEATURE_DRIFT_THRESHOLD = 0.25


# -------------------------------------------------
# Load feature-engineered dataset
# -------------------------------------------------
def load_feature_data(csv_path):
    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=["date_time"],
        )

        logger.info(
            "Monitoring dataset loaded: %d rows, %d columns.",
            df.shape[0],
            df.shape[1],
        )

        return df

    except (FileNotFoundError, pd.errors.ParserError):
        logger.error(
            "Unable to load monitoring dataset: %s",
            csv_path,
            exc_info=True,
        )
        return None


# -------------------------------------------------
# Load deployed Random Forest artifact
# -------------------------------------------------
def load_model_artifact(model_path):
    try:
        artifact = joblib.load(model_path)

        logger.info(
            "Deployment model artifact loaded: %s",
            model_path,
        )

        return artifact

    except (FileNotFoundError, OSError, KeyError):
        logger.error(
            "Unable to load deployment model artifact: %s",
            model_path,
            exc_info=True,
        )
        return None


# -------------------------------------------------
# Prepare same model features used during training
# -------------------------------------------------
def prepare_model_features(df, feature_columns):
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

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        logger.error(
            "Required model features are missing: %s",
            missing_columns,
        )
        return None

    return df


# -------------------------------------------------
# Check prediction error drift
# -------------------------------------------------
def check_prediction_error_drift(
    df,
    model,
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

    baseline_predictions = model.predict(
        X_test
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions,
    )

    # Simulate recently observed production data
    recent_count = int(
        len(df) * 0.20
    )

    recent_df = (
        df.sort_values("date_time")
        .tail(recent_count)
    )

    X_recent = recent_df[
        feature_columns
    ]

    y_recent = recent_df[
        "traffic_volume"
    ]

    recent_predictions = model.predict(
        X_recent
    )

    recent_mae = mean_absolute_error(
        y_recent,
        recent_predictions,
    )

    error_increase = (
        recent_mae - baseline_mae
    ) / baseline_mae

    logger.info(
        "Baseline prediction MAE: %.2f",
        baseline_mae,
    )

    logger.info(
        "Recent-window prediction MAE: %.2f",
        recent_mae,
    )

    logger.info(
        "Prediction MAE change: %.2f%%",
        error_increase * 100,
    )

    error_alert = (
        error_increase
        > ERROR_INCREASE_THRESHOLD
    )

    if error_alert:
        logger.warning(
            "Prediction error drift detected. "
            "MAE increased by more than %.0f%%.",
            ERROR_INCREASE_THRESHOLD * 100,
        )
    else:
        logger.info(
            "Prediction error remains within "
            "the monitoring threshold."
        )

    return (
        baseline_mae,
        recent_mae,
        error_increase,
        error_alert,
        X_train,
        recent_df,
    )


# -------------------------------------------------
# Check feature distribution drift
# -------------------------------------------------
def check_feature_drift(
    reference_data,
    recent_data,
):
    drift_features = [
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

    drift_results = {}
    drift_alerts = []

    for feature in drift_features:
        reference_mean = (
            reference_data[feature].mean()
        )

        recent_mean = (
            recent_data[feature].mean()
        )

        reference_std = (
            reference_data[feature].std()
        )

        if (
            pd.isna(reference_std)
            or reference_std == 0
        ):
            logger.warning(
                "Feature drift check skipped for %s "
                "because reference standard deviation "
                "is zero or unavailable.",
                feature,
            )

            continue

        standardized_difference = abs(
            recent_mean - reference_mean
        ) / reference_std

        drift_results[feature] = (
            standardized_difference
        )

        logger.info(
            "Feature drift %s -> %.4f",
            feature,
            standardized_difference,
        )

        if (
            standardized_difference
            > FEATURE_DRIFT_THRESHOLD
        ):
            drift_alerts.append(feature)

            logger.warning(
                "Feature drift detected: %s "
                "(standardized difference %.4f)",
                feature,
                standardized_difference,
            )

    feature_alert = (
        len(drift_alerts) > 0
    )

    return (
        drift_results,
        drift_alerts,
        feature_alert,
    )


# -------------------------------------------------
# Save monitoring status
# -------------------------------------------------
def save_monitoring_status(
    baseline_mae,
    recent_mae,
    error_increase,
    error_alert,
    drift_results,
    drift_alerts,
    overall_status,
):
    output_path = Path(
        "part3_machine_learning/mlops/"
        "monitoring_status.json"
    )

    output = {
        "status": overall_status,
        "prediction_monitoring": {
            "baseline_mae": round(
                float(baseline_mae),
                2,
            ),
            "recent_mae": round(
                float(recent_mae),
                2,
            ),
            "mae_change_percent": round(
                float(error_increase * 100),
                2,
            ),
            "alert": bool(error_alert),
        },
        "feature_monitoring": {
            "drift_threshold": (
                FEATURE_DRIFT_THRESHOLD
            ),
            "drift_scores": {
                feature: round(
                    float(score),
                    4,
                )
                for feature, score
                in drift_results.items()
            },
            "alert_features": drift_alerts,
        },
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=4,
        )

    logger.info(
        "Monitoring status saved: %s",
        output_path,
    )


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    csv_path = Path(
        "data/processed/traffic_features_part2.csv"
    )

    model_path = Path(
        "part3_machine_learning/models/"
        "random_forest_regressor.joblib"
    )

    df = load_feature_data(csv_path)

    if df is None:
        sys.exit(1)

    artifact = load_model_artifact(
        model_path
    )

    if artifact is None:
        sys.exit(1)

    model = artifact["model"]

    feature_columns = artifact[
        "feature_columns"
    ]

    df = prepare_model_features(
        df,
        feature_columns,
    )

    if df is None:
        sys.exit(1)

    (
        baseline_mae,
        recent_mae,
        error_increase,
        error_alert,
        reference_data,
        recent_data,
    ) = check_prediction_error_drift(
        df,
        model,
        feature_columns,
    )

    (
        drift_results,
        drift_alerts,
        feature_alert,
    ) = check_feature_drift(
        reference_data,
        recent_data,
    )

    if error_alert or feature_alert:
        overall_status = (
            "ALERT / Requires investigation"
        )

        logger.warning(
            "Monitoring status: %s",
            overall_status,
        )

    else:
        overall_status = (
            "PASS / Normal"
        )

        logger.info(
            "Monitoring status: %s",
            overall_status,
        )

    save_monitoring_status(
        baseline_mae,
        recent_mae,
        error_increase,
        error_alert,
        drift_results,
        drift_alerts,
        overall_status,
    )

    # Direct user-facing monitoring result
    print(
        f"Monitoring status: {overall_status}"
    )

    if drift_alerts:
        print(
            "Feature drift detected in: "
            + ", ".join(drift_alerts)
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

    console_handler = (
        logging.StreamHandler()
    )

    console_handler.setFormatter(
        formatter
    )

    file_handler = logging.FileHandler(
        "part3_machine_learning/part3.log"
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

    main()