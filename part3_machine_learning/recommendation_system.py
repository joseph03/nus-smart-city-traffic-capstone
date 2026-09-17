import logging
import sys
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


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
# Identify suitable travel windows
# -------------------------------------------------
def recommend_travel_window(
    df,
    day_type,
    weather_main=None,
):
    df = df.copy()

    day_type = day_type.lower()

    if day_type == "weekday":
        filtered = df[df["weekend"] == 0]

    elif day_type == "weekend":
        filtered = df[df["weekend"] == 1]

    else:
        logger.error(
            "Invalid day type: %s",
            day_type,
        )

        return None

    # Apply weather filter if supplied
    if weather_main is not None:
        weather_matches = filtered[
            filtered["weather_main"]
            .str.lower()
            .eq(weather_main.lower())
        ]

        if weather_matches.empty:
            logger.warning(
                "No records found for weather condition: %s. "
                "Using day-type data only.",
                weather_main,
            )
        else:
            filtered = weather_matches

    if filtered.empty:
        logger.error(
            "No records available for recommendation."
        )
        return None

    filtered = filtered[
        filtered["hour"].between(
            6,
            21,
        )
    ]
    
    hourly_average = (
        filtered.groupby("hour")[
            "traffic_volume"
        ]
        .mean()
        .sort_values()
    )

    recommended_hour = int(
        hourly_average.index[0]
    )

    expected_volume = float(
        hourly_average.iloc[0]
    )

    logger.info(
        "Recommended travel hour: %02d:00",
        recommended_hour,
    )

    logger.info(
        "Expected average traffic volume: %.2f",
        expected_volume,
    )

    return {
        "hour": recommended_hour,
        "traffic_volume": expected_volume,
    }


# -------------------------------------------------
# Generate user-facing recommendation
# -------------------------------------------------
def generate_recommendation(
    day_type,
    recommendation,
    weather_main=None,
):
    if recommendation is None:
        return None

    hour = recommendation["hour"]
    expected_volume = (
        recommendation["traffic_volume"]
    )

    next_hour = (hour + 1) % 24

    if weather_main is None:
        weather_text = ""
    else:
        weather_text = (
            f" under {weather_main} weather"
        )

    message = (
        f"For a {day_type.lower()} journey"
        f"{weather_text}, consider travelling "
        f"between {hour:02d}:00 and "
        f"{next_hour:02d}:00. "
        f"Historical traffic volume during "
        f"this period averages approximately "
        f"{expected_volume:.0f} vehicles."
    )

    return message


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

    # Initial demonstration
    day_type = "weekday"
    weather_main = None

    recommendation = recommend_travel_window(
        df,
        day_type,
        weather_main,
    )

    message = generate_recommendation(
        day_type,
        recommendation,
        weather_main,
    )

    if message is not None:
        print(message)


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