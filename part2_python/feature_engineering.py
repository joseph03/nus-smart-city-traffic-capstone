import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------------------------------------
# Load cleaned dataset
# -------------------------------------------------
def load_cleaned_data(csv_path):
    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=["date_time"]
        )

        logger.info(
            "Cleaned dataset loaded for feature engineering: "
            "%d rows, %d columns.",
            df.shape[0],
            df.shape[1],
        )

        return df

    except (FileNotFoundError, pd.errors.ParserError):
        logger.error(
            "Unable to load cleaned dataset: %s",
            csv_path,
            exc_info=True,
        )
        return None


# -------------------------------------------------
# Create time features
# -------------------------------------------------
def create_time_features(df):
    df = df.copy()

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["weekend"] = (df["day_of_week"] >= 5).astype(int)

    logger.info(
        "Time features created: hour, day_of_week, weekend."
    )

    return df

# -------------------------------------------------
# Create cyclical time features
# -------------------------------------------------
def create_cyclical_features(df):
    df = df.copy()

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    logger.info(
        "Cyclical hour features created: hour_sin, hour_cos."
    )

    return df

# -------------------------------------------------
# Create weather features
# -------------------------------------------------
def create_weather_features(df):
    df = df.copy()

    df["is_clear"] = (
        df["weather_main"]
        .str.lower()
        .eq("clear")
        .astype(int)
    )

    df["has_rain"] = (
        df["rain_1h"] > 0
    ).astype(int)

    logger.info(
        "Weather features created: is_clear, has_rain."
    )

    return df

# -------------------------------------------------
# Encode weather category
# -------------------------------------------------
def encode_weather_category(df):
    df = df.copy()

    weather_encoded = pd.get_dummies(
        df["weather_main"],
        prefix="weather",
        dtype=int,
    )

    df = pd.concat(
        [df, weather_encoded],
        axis=1,
    )

    logger.info(
        "weather_main encoded into %d one-hot columns.",
        weather_encoded.shape[1],
    )

    return df

# -------------------------------------------------
# Scale numerical features
# -------------------------------------------------
def scale_numerical_features(df):
    df = df.copy()

    for column in ["temp", "traffic_volume"]:
        mean_value = df[column].mean()
        std_value = df[column].std()

        df[f"{column}_scaled"] = (
            df[column] - mean_value
        ) / std_value

        logger.debug(
            "%s scaling values: mean=%.4f, std=%.4f",
            column,
            mean_value,
            std_value,
        )

    logger.info(
        "Numerical features scaled: temp, traffic_volume."
    )

    return df

# -------------------------------------------------
# Create congestion category
# -------------------------------------------------
def create_congestion_category(df):
    df = df.copy()

    q1 = df["traffic_volume"].quantile(0.25)
    q3 = df["traffic_volume"].quantile(0.75)

    logger.debug(
        "Congestion thresholds calculated: Q1=%.2f, Q3=%.2f",
        q1,
        q3,
    )

    df["congestion_category"] = pd.cut(
        df["traffic_volume"],
        bins=[-np.inf, q1, q3, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    logger.info(
        "Data-driven congestion category created using "
        "traffic-volume quartiles."
    )

    return df

# -------------------------------------------------
# Save feature-engineered dataset
# -------------------------------------------------
def save_feature_data(df, output_path):
    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            output_path,
            index=False
        )

        logger.info(
            "Feature-engineered dataset saved successfully: %s",
            output_path,
        )

    except OSError:
        logger.error(
            "Failed to save feature-engineered dataset: %s",
            output_path,
            exc_info=True,
        )
        return False

    return True

def main():
    csv_path = Path(
        "data/processed/traffic_clean_part2.csv"
    )

    df = load_cleaned_data(csv_path)

    if df is None:
        sys.exit(1)

    logger.info(
        "Dataset shape before feature engineering: %d rows, %d columns.",
        df.shape[0],
        df.shape[1],
    )

    df = create_time_features(df)
    df = create_cyclical_features(df)
    df = create_weather_features(df)
    df = encode_weather_category(df)
    df = scale_numerical_features(df)
    df = create_congestion_category(df)

    logger.info(
        "Dataset shape after feature engineering: "
        "%d rows, %d columns.",
        df.shape[0],
        df.shape[1],
    )

    output_path = Path(
        "data/processed/traffic_features_part2.csv"
    )

    if not save_feature_data(df, output_path):
        sys.exit(1)

if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        "part2_python/pipeline.log"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    main()
    