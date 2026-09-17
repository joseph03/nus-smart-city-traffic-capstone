import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# for building classification model, and the use of logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    # for regression
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# second classification model: Random Forest.
from sklearn.ensemble import RandomForestClassifier

# regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression   # LogisticRegression is needed too

logger = logging.getLogger(__name__)


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
            "Feature-engineered dataset loaded: %d rows, %d columns.",
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
# Create Part 3 congestion category
# -------------------------------------------------
def create_part3_congestion_category(df):
    df = df.copy()

    q1, q2, q3 = df["traffic_volume"].quantile(
        [0.25, 0.50, 0.75]
    ).values

    logger.info(
        "Part 3 congestion thresholds: "
        "Q1=%.2f, Q2=%.2f, Q3=%.2f",
        q1,
        q2,
        q3,
    )

    def bucket(value):
        if value <= q1:
            return "Low"
        elif value <= q2:
            return "Medium"
        elif value <= q3:
            return "High"
        else:
            return "Severe"

    df["congestion_category_part3"] = (
        df["traffic_volume"].apply(bucket)
    )

    logger.info(
        "Part 3 congestion category created: "
        "Low, Medium, High, Severe."
    )

    return df


# -------------------------------------------------
# Create proxy accident-risk label
# -------------------------------------------------
def create_high_risk_label(df):
    df = df.copy()

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
        df["weather_main"].isin(SEVERE_WEATHER)
        | (df["is_low_visibility"] == 1)
    )

    df["high_risk"] = (
        high_congestion & risky_weather
    ).astype(int)

    high_risk_count = int(df["high_risk"].sum())
    total_count = len(df)
    low_risk_count = total_count - high_risk_count

    logger.info(
        "Proxy high_risk label created."
    )

    logger.info(
        "high_risk = 1 -> %d records",
        high_risk_count,
    )

    logger.info(
        "high_risk = 0 -> %d records",
        low_risk_count,
    )

    return df


# -------------------------------------------------
# Prepare common machine-learning feature set
# -------------------------------------------------
def prepare_model_features(df):
    df = df.copy()

    # Holiday flag
    df["holiday_flag"] = (
        df["holiday"]
        .str.lower()
        .ne("none")
        .astype(int)
    )

    # Cyclical day-of-week encoding
    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # Core numerical / engineered features
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

    # Weather one-hot columns created in Part 2
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
        "Common ML feature set prepared with %d features.",
        len(feature_columns),
    )

    logger.debug(
        "ML features: %s",
        feature_columns,
    )

    return df, feature_columns

# -------------------------------------------------
# Train and evaluate Logistic Regression classifier
# -------------------------------------------------
def train_logistic_classifier(df, feature_columns):
    X = df[feature_columns]
    y = df["high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    logger.info(
        "Classification data split: "
        "%d training records, %d test records.",
        len(X_train),
        len(X_test),
    )

    train_high_risk = int(y_train.sum())
    train_low_risk = len(y_train) - train_high_risk

    test_high_risk = int(y_test.sum())
    test_low_risk = len(y_test) - test_high_risk

    logger.info(
        "Training high_risk = 1 -> %d records",
        train_high_risk,
    )

    logger.info(
        "Training low_risk = 0 -> %d records",
        train_low_risk,
    )

    logger.info(
        "Test high_risk = 1 -> %d records",
        test_high_risk,
    )

    logger.info(
        "Test low_risk = 0 -> %d records",
        test_low_risk,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    logger.info(
        "Logistic Regression classification results:"
    )

    logger.info(
        "Accuracy: %.4f",
        accuracy,
    )

    logger.info(
        "Precision: %.4f",
        precision,
    )

    logger.info(
        "Recall: %.4f",
        recall,
    )

    logger.info(
        "F1-score: %.4f",
        f1,
    )

    logger.info(
        "ROC AUC: %.4f",
        roc_auc,
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }

    return model, metrics

# -------------------------------------------------
# Train and evaluate Random Forest classifier
# -------------------------------------------------
def train_random_forest_classifier(df, feature_columns):
    X = df[feature_columns]
    y = df["high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    logger.info(
        "Random Forest classification results:"
    )

    logger.info(
        "Accuracy: %.4f",
        accuracy,
    )

    logger.info(
        "Precision: %.4f",
        precision,
    )

    logger.info(
        "Recall: %.4f",
        recall,
    )

    logger.info(
        "F1-score: %.4f",
        f1,
    )

    logger.info(
        "ROC AUC: %.4f",
        roc_auc,
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }

    return model, metrics

# -------------------------------------------------
# Train and evaluate Linear Regression
# -------------------------------------------------
def train_linear_regression(df, feature_columns):
    X = df[feature_columns]
    y = df["traffic_volume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    logger.info(
        "Regression data split: "
        "%d training records, %d test records.",
        len(X_train),
        len(X_test),
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    logger.info(
        "Linear Regression results:"
    )

    logger.info(
        "MAE: %.2f",
        mae,
    )

    logger.info(
        "R-squared: %.4f",
        r2,
    )

    metrics = {
        "mae": mae,
        "r2": r2,
    }

    return model, metrics

# -------------------------------------------------
# Train and evaluate Random Forest Regression
# -------------------------------------------------
def train_random_forest_regression(df, feature_columns):
    X = df[feature_columns]
    y = df["traffic_volume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    logger.info(
        "Random Forest Regression results:"
    )

    logger.info(
        "MAE: %.2f",
        mae,
    )

    logger.info(
        "R-squared: %.4f",
        r2,
    )

    metrics = {
        "mae": mae,
        "r2": r2,
    }

    return model, metrics

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

    df = create_part3_congestion_category(df)

    df = create_high_risk_label(df)

    df, feature_columns = prepare_model_features(df)

    logistic_model, logistic_metrics = train_logistic_classifier(
        df,
        feature_columns,
    )

    random_forest_model, random_forest_metrics = (
        train_random_forest_classifier(
            df,
            feature_columns,
        )
    )

    linear_regression_model, linear_regression_metrics = (
        train_linear_regression(
            df,
            feature_columns,
        )
    )

    random_forest_regression_model, random_forest_regression_metrics = (
        train_random_forest_regression(
            df,
            feature_columns,
        )
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