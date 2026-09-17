import logging
import sys
from pathlib import Path

import pandas as pd


# -------------------------------------------------
# Logging setup
# -------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
     "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler("part2_python/pipeline.log")
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# -------------------------------------------------
# Expected schema
# -------------------------------------------------
EXPECTED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume",
]


# -------------------------------------------------
# Load raw CSV
# -------------------------------------------------
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)

        logger.info(
            "Raw CSV loaded successfully: %d rows, %d columns",
            df.shape[0],
            df.shape[1],
        )

        return df

    except FileNotFoundError:
        logger.error(
            "CSV file not found: %s",
            csv_path,
            exc_info=True,
        )
        return None

    except pd.errors.ParserError:
        logger.error(
            "CSV parsing error: %s",
            csv_path,
            exc_info=True,
        )
        return None

# -------------------------------------------------
# Validate schema
# -------------------------------------------------
def validate_schema(df):
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        logger.error(
            "Schema validation failed. Missing columns: %s",
            missing_columns,
        )
        return False

    logger.info(
        "Schema validation successful. All expected columns are present."
    )

    return True

# -------------------------------------------------
# Standardise categorical values
# -------------------------------------------------
def standardise_categories(df):
    df = df.copy()

    df["holiday"] = (
        df["holiday"]
        .fillna("None")
        .astype(str)
        .str.strip()
    )

    df["weather_main"] = (
        df["weather_main"]
        .astype(str)
        .str.strip()
    )

    df["weather_description"] = (
        df["weather_description"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    logger.info(
        "Categorical values standardised for holiday, "
        "weather_main and weather_description."
    )

    return df

# -------------------------------------------------
# Parse and validate date/time
# -------------------------------------------------
def parse_datetime(df):
    df = df.copy()

    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )

    invalid_dates = df["date_time"].isna().sum()

    if invalid_dates > 0:
        logger.warning(
            "%d rows contain invalid date_time values and will be dropped.",
            invalid_dates,
        )

        df = df.dropna(subset=["date_time"])
    else:
        logger.info(
            "All date_time values parsed successfully."
        )

    return df

# -------------------------------------------------
# Remove duplicate rows
# -------------------------------------------------
def remove_duplicates(df):
    df = df.copy()

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates()

        logger.warning(
            "%d duplicate rows were removed.",
            duplicate_count,
        )
    else:
        logger.info(
            "No duplicate rows found."
        )

    return df

# -------------------------------------------------
# Handle invalid temperature values
# -------------------------------------------------
def handle_invalid_temperature(df):
    df = df.copy()

    invalid_temp = df["temp"] <= 0
    invalid_count = invalid_temp.sum()

    if invalid_count == 0:
        logger.info(
            "No invalid temperature values found."
        )
        return df

    for month in df.loc[invalid_temp, "date_time"].dt.month.unique():

        month_valid_temp = df.loc[
            (df["date_time"].dt.month == month)
            & (df["temp"] > 0),    # exclude 0k
            "temp",
        ]

        month_median = month_valid_temp.median()

        rows_to_fix = (
            invalid_temp
            & (df["date_time"].dt.month == month)
        )

        affected_rows = rows_to_fix.sum()

        df.loc[rows_to_fix, "temp"] = month_median

        logger.warning(
            "%d invalid temperature rows in month %d "
            "were imputed with monthly median %.2f K.",
            affected_rows,
            month,
            month_median,
        )

    return df

# -------------------------------------------------
# Handle extreme rainfall values
# -------------------------------------------------
def handle_extreme_rainfall(df):
    df = df.copy()

    extreme_rain = df["rain_1h"] >= 9000
    extreme_count = extreme_rain.sum()

    if extreme_count == 0:
        logger.info(
            "No extreme rainfall values found."
        )
        return df

    for month in df.loc[extreme_rain, "date_time"].dt.month.unique():

        month_valid_rain = df.loc[
            (df["date_time"].dt.month == month)
            & (df["rain_1h"] > 0)
            & (df["rain_1h"] < 9000),
            "rain_1h",
        ]

        month_median = month_valid_rain.median()

        rows_to_fix = (
            extreme_rain
            & (df["date_time"].dt.month == month)
        )

        affected_rows = rows_to_fix.sum()

        df.loc[rows_to_fix, "rain_1h"] = month_median

        logger.warning(
            "%d extreme rainfall rows in month %d "
            "were imputed with monthly positive-rain median %.2f mm.",
            affected_rows,
            month,
            month_median,
        )

    return df

# -------------------------------------------------
# Save cleaned dataset
# -------------------------------------------------
def save_cleaned_data(df, output_path):
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
            "Cleaned dataset saved successfully: %s",
            output_path,
        )

    except OSError:
        logger.error(
            "Failed to save cleaned dataset: %s",
            output_path,
            exc_info=True,
        )
        return False

    return True

# -------------------------------------------------
# Validate cleaned dataset
# -------------------------------------------------
def validate_cleaned_data(df):
    duplicate_count = df.duplicated().sum()
    invalid_temp_count = (df["temp"] <= 0).sum()
    extreme_rain_count = (df["rain_1h"] >= 9000).sum()

    if duplicate_count > 0:
        logger.error(
            "Validation failed: %d duplicate rows remain.",
            duplicate_count,
        )
        return False

    if invalid_temp_count > 0:
        logger.error(
            "Validation failed: %d invalid temperature values remain.",
            invalid_temp_count,
        )
        return False

    if extreme_rain_count > 0:
        logger.error(
            "Validation failed: %d extreme rainfall values remain.",
            extreme_rain_count,
        )
        return False

    logger.info(
        "Cleaned data validation successful: %d rows, %d columns.",
        df.shape[0],
        df.shape[1],
    )

    return True

def main():
    csv_path = Path("data/raw/Metro_Interstate_Traffic_Volume.csv")

    df = load_data(csv_path)

    if df is None:
        sys.exit(1)

    if not validate_schema(df):
        sys.exit(1)

    df = standardise_categories(df)
    df = parse_datetime(df)
    df = remove_duplicates(df)
    df = handle_invalid_temperature(df)
    df = handle_extreme_rainfall(df)

    if not validate_cleaned_data(df):
        sys.exit(1)

    output_path = Path(
    "data/processed/traffic_clean_part2.csv")

    if not save_cleaned_data(df, output_path):
        sys.exit(1)


if __name__ == "__main__":
    main()