import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


logger = logging.getLogger(__name__)


# -------------------------------------------------
# Load feature-engineered dataset
# -------------------------------------------------
def load_feature_data(csv_path):
    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=["date_time"]
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
# Visualisation 1: Traffic demand by hour
# -------------------------------------------------
def plot_traffic_by_hour(df, output_path):
    hourly_traffic = (
        df.groupby("hour")["traffic_volume"]
        .mean()
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        hourly_traffic.index,
        hourly_traffic.values,
        marker="o"
    )

    plt.title("Average Traffic Volume by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Traffic Volume")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    logger.info(
        "Figure saved successfully: %s",
        output_path,
    )

# -------------------------------------------------
# Visualisation 2: Weekday vs weekend traffic
# -------------------------------------------------
def plot_weekday_vs_weekend(df, output_path):
    traffic_comparison = (
        df.groupby("weekend")["traffic_volume"]
        .mean()
    )

    labels = ["Weekday", "Weekend"]

    plt.figure(figsize=(8, 6))

    plt.bar(
        labels,
        traffic_comparison.values
    )

    plt.title("Average Traffic Volume: Weekday vs Weekend")
    plt.xlabel("Day Type")
    plt.ylabel("Average Traffic Volume")
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    logger.info(
        "Figure saved successfully: %s",
        output_path,
    )

# -------------------------------------------------
# Visualisation 3: Traffic by weather condition
# -------------------------------------------------
def plot_traffic_by_weather(df, output_path):
    weather_traffic = (
        df.groupby("weather_main")["traffic_volume"]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        weather_traffic.index,
        weather_traffic.values
    )

    plt.title("Average Traffic Volume by Weather Condition")
    plt.xlabel("Average Traffic Volume")
    plt.ylabel("Weather Condition")
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    logger.info(
        "Figure saved successfully: %s",
        output_path,
    )

def main():
    csv_path = Path(
        "data/processed/traffic_features_part2.csv"
    )

    df = load_feature_data(csv_path)

    if df is None:
        sys.exit(1)

    figures_dir = Path("part2_python/figures")
    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_traffic_by_hour(
        df,
        figures_dir / "task3_1_traffic_by_hour.png"
    )

    plot_weekday_vs_weekend(
       df,
        figures_dir / "task3_2_weekday_vs_weekend.png"
    )

    plot_traffic_by_weather(
        df,
        figures_dir / "task3_3_traffic_by_weather.png"
    )


if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
         "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
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