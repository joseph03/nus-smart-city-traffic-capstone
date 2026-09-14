import sqlite3
import pandas as pd

csv_file = "data/raw/Metro_Interstate_Traffic_Volume.csv"
db_file = "part1_data_analytics/traffic.db"

# --------------------------------------------------
# 1. Load raw data
# --------------------------------------------------
df = pd.read_csv(csv_file)

print("Raw rows:", len(df))

# --------------------------------------------------
# 2. Parse date_time
# --------------------------------------------------
df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

invalid_dates = df["date_time"].isna().sum()
print("Invalid dates:", invalid_dates)

# --------------------------------------------------
# 3. Standardise holiday values
# --------------------------------------------------
df["holiday"] = df["holiday"].fillna("None")

# --------------------------------------------------
# 4. Standardise categorical text
# --------------------------------------------------
df["weather_main"] = df["weather_main"].str.strip()
df["weather_description"] = (
    df["weather_description"]
    .str.strip()
    .str.lower()
)

# --------------------------------------------------
# 5. Remove duplicate rows
# --------------------------------------------------
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates().copy()

print("Duplicate rows removed:", duplicates_before)

# --------------------------------------------------
# 6. Replace invalid temperature values
# --------------------------------------------------
invalid_temp = df["temp"] <= 0
print("Invalid temperature rows:", invalid_temp.sum())

# Calculate monthly medians using only valid temperatures
valid_temp = df[df["temp"] > 0]

monthly_temp_median = (
    valid_temp
    .groupby(valid_temp["date_time"].dt.month)["temp"]
    .median()
)

for month in monthly_temp_median.index:
    condition = (
        (df["date_time"].dt.month == month)
        & (df["temp"] <= 0)
    )

    count = condition.sum()

    if count > 0:
        replacement = monthly_temp_median.loc[month]

        print(
            f"Replacing {count} invalid temperature values "
            f"in month {month} with median {replacement}"
        )

        df.loc[condition, "temp"] = replacement

# --------------------------------------------------
# 7. Replace extreme rainfall value
# --------------------------------------------------

# Treat rainfall >= 9000 mm/hour as invalid
extreme_rain = df["rain_1h"] >= 9000

print("Extreme rainfall rows:", extreme_rain.sum())

for index in df[extreme_rain].index:

    month = df.loc[index, "date_time"].month

    valid_rain = df[
        (df["date_time"].dt.month == month)
        & (df["rain_1h"] > 0)
        & (df["rain_1h"] < 9000)
    ]

    replacement = valid_rain["rain_1h"].median()

    print(
        f"Replacing rainfall value {df.loc[index, 'rain_1h']} "
        f"on {df.loc[index, 'date_time']} "
        f"with median {replacement}"
    )

    df.loc[index, "rain_1h"] = replacement

# --------------------------------------------------
# 8. Final validation
# --------------------------------------------------
print("\n=== CLEANED DATA CHECK ===")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("Missing values:")
print(df.isna().sum())

print("Duplicate rows:", df.duplicated().sum())

print("Temperature <= 0:", (df["temp"] <= 0).sum())

print(
    "Extreme rainfall >= 9000:",
    (df["rain_1h"] >= 9000).sum()
)

print(
    "Cloud cover outside 0-100:",
    ((df["clouds_all"] < 0) |
     (df["clouds_all"] > 100)).sum()
)

print(
    "Negative traffic volume:",
    (df["traffic_volume"] < 0).sum()
)

# --------------------------------------------------
# 9. Save cleaned CSV
# --------------------------------------------------
clean_csv = "data/processed/traffic_clean.csv"

df.to_csv(clean_csv, index=False)

print("\nCleaned CSV saved to:", clean_csv)

# --------------------------------------------------
# 10. Save cleaned table into SQLite
# --------------------------------------------------
conn = sqlite3.connect(db_file)

df.to_sql(
    "traffic_clean",
    conn,
    if_exists="replace",
    index=False
)

# Verify SQLite table
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM traffic_clean")
row_count = cursor.fetchone()[0]

print("Rows in traffic_clean table:", row_count)

conn.close()
