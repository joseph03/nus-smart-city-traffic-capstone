import pandas as pd

csv_file = "data/raw/Metro_Interstate_Traffic_Volume.csv"

df = pd.read_csv(csv_file)

print("=== BASIC INFORMATION ===")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== MISSING VALUES ===")
print(df.isna().sum())

print("\n=== DUPLICATE ROWS ===")
print(df.duplicated().sum())

print("\n=== NUMERICAL SUMMARY ===")
print(df.describe())

print("\n=== DATE CHECK ===")
dates = pd.to_datetime(df["date_time"], errors="coerce")
print("Earliest date:", dates.min())
print("Latest date:", dates.max())
print("Invalid dates:", dates.isna().sum())

print("\n=== TEMPERATURE CHECK ===")
print("Minimum temp:", df["temp"].min())
print("Maximum temp:", df["temp"].max())
print("Rows with temp <= 0:", (df["temp"] <= 0).sum())

print("\n=== RAIN CHECK ===")
print("Minimum rain_1h:", df["rain_1h"].min())
print("Maximum rain_1h:", df["rain_1h"].max())
print("Rows with negative rain:", (df["rain_1h"] < 0).sum())

print("\n=== SNOW CHECK ===")
print("Minimum snow_1h:", df["snow_1h"].min())
print("Maximum snow_1h:", df["snow_1h"].max())
print("Rows with negative snow:", (df["snow_1h"] < 0).sum())

print("\n=== CLOUD COVER CHECK ===")
print("Minimum clouds_all:", df["clouds_all"].min())
print("Maximum clouds_all:", df["clouds_all"].max())
print("Outside 0-100:",
      ((df["clouds_all"] < 0) | (df["clouds_all"] > 100)).sum())

print("\n=== TRAFFIC VOLUME CHECK ===")
print("Minimum traffic_volume:", df["traffic_volume"].min())
print("Maximum traffic_volume:", df["traffic_volume"].max())
print("Negative traffic volume:", (df["traffic_volume"] < 0).sum())

print("\n=== HOLIDAY VALUES ===")
print(df["holiday"].value_counts(dropna=False))

print("\n=== WEATHER MAIN VALUES ===")
print(df["weather_main"].value_counts(dropna=False))

print("\n=== WEATHER DESCRIPTION VALUES ===")
print(df["weather_description"].value_counts(dropna=False))
