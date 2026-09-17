import logging
import sys
from pathlib import Path

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# rules
from mlxtend.frequent_patterns import (
    apriori,
    association_rules,
)


logger = logging.getLogger(__name__)


# -------------------------------------------------
# Weather groups for clustering
# -------------------------------------------------
LOW_VISIBILITY_WEATHER = [
    "Fog",
    "Mist",
    "Smoke",
    "Haze",
]

SEVERE_WEATHER = [
    "Thunderstorm",
    "Snow",
    "Squall",
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
# Create numerical weather severity for clustering
# -------------------------------------------------
def create_weather_severity(df):
    df = df.copy()

    df["weather_severity"] = 0

    df.loc[
        df["weather_main"].isin(["Rain", "Drizzle"]),
        "weather_severity",
    ] = 1

    df.loc[
        df["weather_main"].isin(LOW_VISIBILITY_WEATHER),
        "weather_severity",
    ] = 2

    df.loc[
        df["weather_main"].isin(SEVERE_WEATHER),
        "weather_severity",
    ] = 3

    logger.info(
        "Weather severity feature created with levels 0 to 3."
    )

    logger.debug(
        "Weather severity counts: %s",
        df["weather_severity"]
        .value_counts()
        .sort_index()
        .to_dict(),
    )

    return df


# -------------------------------------------------
# Apply K-means clustering
# -------------------------------------------------
def apply_kmeans(df):
    cluster_features = [
        "hour",
        "traffic_volume",
        "weather_severity",
    ]

    X = df[cluster_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10,
    )

    df = df.copy()

    df["traffic_cluster"] = kmeans.fit_predict(
        X_scaled
    )

    logger.info(
        "K-means clustering completed using 4 clusters."
    )

    cluster_counts = (
        df["traffic_cluster"]
        .value_counts()
        .sort_index()
    )

    for cluster, count in cluster_counts.items():
        logger.info(
            "Cluster %d -> %d records",
            cluster,
            count,
        )

    cluster_summary = (
        df.groupby("traffic_cluster")[
            [
                "hour",
                "traffic_volume",
                "weather_severity",
            ]
        ]
        .mean()
        .round(2)
    )

    logger.info(
        "K-means cluster summary:\n%s",
        cluster_summary,
    )

    return df, kmeans, scaler, cluster_summary

# -------------------------------------------------
# Prepare categorical features for association rules
# -------------------------------------------------
def create_association_features(df):
    df = df.copy()

    # Time-of-day category
    df["time_period"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=[
            "Night",
            "Morning",
            "Afternoon",
            "Evening",
        ],
    )

    # Weekday / weekend category
    df["day_type"] = df["weekend"].map(
        {
            0: "Weekday",
            1: "Weekend",
        }
    )

    # Weather category
    df["weather_group"] = "Normal"

    df.loc[
        df["weather_main"].isin(
            ["Rain", "Drizzle"]
        ),
        "weather_group",
    ] = "Rain"

    df.loc[
        df["weather_main"].isin(
            LOW_VISIBILITY_WEATHER
        ),
        "weather_group",
    ] = "LowVisibility"

    df.loc[
        df["weather_main"].isin(
            SEVERE_WEATHER
        ),
        "weather_group",
    ] = "Severe"

    # Part 3 four-level congestion category
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

    df["association_congestion"] = (
        df["traffic_volume"]
        .apply(congestion_bucket)
    )

    logger.info(
        "Association-rule features created."
    )

    logger.info(
        "Association congestion thresholds: "
        "Q1=%.2f, Q2=%.2f, Q3=%.2f",
        q1,
        q2,
        q3,
    )

    return df


# -------------------------------------------------
# Apply association rule mining
# -------------------------------------------------
def apply_association_rules(df):
    basket = pd.concat(
        [
            pd.get_dummies(
                df["time_period"],
                prefix="Time",
                dtype=bool,
            ),
            pd.get_dummies(
                df["day_type"],
                prefix="Day",
                dtype=bool,
            ),
            pd.get_dummies(
                df["weather_group"],
                prefix="Weather",
                dtype=bool,
            ),
            pd.get_dummies(
                df["association_congestion"],
                prefix="Congestion",
                dtype=bool,
            ),
        ],
        axis=1,
    )

    logger.info(
        "Association-rule transaction matrix prepared: "
        "%d records, %d items.",
        basket.shape[0],
        basket.shape[1],
    )

    frequent_itemsets = apriori(
        basket,
        min_support=0.02,
        use_colnames=True,
        max_len=4,
    )

    logger.info(
        "%d frequent itemsets identified.",
        len(frequent_itemsets),
    )

    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=1.0,
    )

    # Keep rules that predict a congestion category
    congestion_rules = rules[
        rules["consequents"].apply(
            lambda items:
            len(items) == 1
            and next(iter(items)).startswith(
                "Congestion_"
            )
        )
        &
        rules["antecedents"].apply(
            lambda items:
            not any(
                str(item).startswith(
                    "Congestion_"
                )
                for item in items
            )
        )
    ].copy()

    congestion_rules = (
        congestion_rules.sort_values(
            by=[
                "lift",
                "confidence",
                "support",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    logger.info(
        "%d association rules predicting congestion "
        "were identified.",
        len(congestion_rules),
    )

    top_rules = congestion_rules.head(10)

    for index, row in top_rules.iterrows():
        antecedents = ", ".join(
            sorted(row["antecedents"])
        )

        consequents = ", ".join(
            sorted(row["consequents"])
        )

        logger.info(
            "Rule %d: %s -> %s | "
            "support=%.4f, confidence=%.4f, lift=%.4f",
            index + 1,
            antecedents,
            consequents,
            row["support"],
            row["confidence"],
            row["lift"],
        )

    return (
        frequent_itemsets,
        congestion_rules,
        top_rules,
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

    df = create_weather_severity(df)

    df, kmeans_model, scaler, cluster_summary = (
        apply_kmeans(df)
    )

    df = create_association_features(df)

    (
        frequent_itemsets,
        congestion_rules,
        top_rules,
    ) = apply_association_rules(df)


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