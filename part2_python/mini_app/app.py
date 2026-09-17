
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)

class TrafficArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        logger.error("Invalid CLI input: %s", message)

        print(
            "Invalid command or arguments.\n"
            "Available commands:\n"
            "  compare-days\n"
            "  high-traffic\n"
            '  traffic-at "YYYY-MM-DD HH:MM"'
        )

        sys.exit(2)

# -------------------------------------------------
# Load processed dataset
# -------------------------------------------------
def load_data(csv_path):
    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=["date_time"]
        )

        logger.info(
            "Processed dataset loaded: %d rows, %d columns.",
            df.shape[0],
            df.shape[1],
        )

        return df

    except (FileNotFoundError, pd.errors.ParserError):
        logger.error(
            "Unable to load processed dataset: %s",
            csv_path,
            exc_info=True,
        )
        return None


# -------------------------------------------------
# Command 1: Compare weekday and weekend traffic
# -------------------------------------------------
def compare_weekday_weekend(df):
    averages = (
        df.groupby("weekend")["traffic_volume"]
        .mean()
    )

    weekday = averages.get(0, float("nan"))
    weekend = averages.get(1, float("nan"))

    print(
        f"Average weekday traffic volume: {weekday:.0f} vehicles"
    )

    print(
        f"Average weekend traffic volume: {weekend:.0f} vehicles"
    )

# -------------------------------------------------
# Command 2: Identify high-traffic hours
# -------------------------------------------------
def show_high_traffic_hours(df):
    hourly_traffic = (
        df.groupby("hour")["traffic_volume"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )

    print("Top 5 highest-traffic hours:")

    for hour, volume in hourly_traffic.items():
        print(
            f"{hour:02d}:00 - average traffic volume: "
            f"{volume:.0f} vehicles"
        )

# -------------------------------------------------
# Command 3: Query traffic for a specific date/time
# -------------------------------------------------
def traffic_at_datetime(df, date_time_value):
    try:
        query_time = pd.to_datetime(
            date_time_value,
            format="%Y-%m-%d %H:%M"
        )

    except ValueError:
        logger.error(
            "Invalid date/time format: %s. "
            "Expected format: YYYY-MM-DD HH:MM",
            date_time_value,
        )

        print(
            "Invalid date/time format. "
            "Use YYYY-MM-DD HH:MM"
        )

        return

    matches = df[
        df["date_time"] == query_time
    ]

    if matches.empty:
        print(
            f"No traffic record found for "
            f"{query_time:%Y-%m-%d %H:%M}."
        )

        return

    print(
        f"Traffic information for "
        f"{query_time:%Y-%m-%d %H:%M}:"
    )

    for _, row in matches.iterrows():
        print(
            f"Traffic volume: {row['traffic_volume']} vehicles"
        )
        print(
            f"Weather: {row['weather_main']}"
        )
        print(
            f"Temperature: {row['temp']:.2f} K"
        )

# -------------------------------------------------
# Command-line interface
# -------------------------------------------------
def main():
    #parser = argparse.ArgumentParser(
    #    description="Mini Traffic Analytics Application"
    #)

    parser = TrafficArgumentParser(
        description="Mini Traffic Analytics Application"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "compare-days",
        help="Compare average weekday and weekend traffic.",
    )

    subparsers.add_parser(
        "high-traffic",
        help="Show the five hours with the highest average traffic.",
    )

    traffic_at_parser = subparsers.add_parser(
        "traffic-at",
        help="Query traffic for a specific date and time.",
    )

    traffic_at_parser.add_argument(
        "datetime",
        help="Date/time in YYYY-MM-DD HH:MM format.",
    )

    args = parser.parse_args()

    logger.info(
        "CLI command invoked: %s",
        args.command,
    )

    csv_path = Path(
        "data/processed/traffic_features_part2.csv"
    )

    df = load_data(csv_path)

    if df is None:
        sys.exit(1)

    if args.command == "compare-days":
        compare_weekday_weekend(df)
    elif args.command == "high-traffic":
        show_high_traffic_hours(df)
    elif args.command == "traffic-at":
        logger.info(
          "traffic-at argument: %s",
          args.datetime,
        )

        traffic_at_datetime(
          df,
          args.datetime,
        )


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - "
        "%(funcName)s - %(message)s"
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