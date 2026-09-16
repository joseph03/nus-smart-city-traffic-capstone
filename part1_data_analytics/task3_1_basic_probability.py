import pandas as pd

df = pd.read_csv("data/processed/traffic_clean.csv")

total_rows = len(df)

congestion = df["traffic_volume"] > 5500
clear_weather = df["weather_main"] == "Clear"

p_congestion = congestion.sum() / total_rows
p_clear = clear_weather.sum() / total_rows
p_congestion_and_clear = (congestion & clear_weather).sum() / total_rows

print("=== TASK 3.1 BASIC PROBABILITY ===")
print("Total records:", total_rows)
print("P(Congestion):", round(p_congestion, 4))
print("P(Clear Weather):", round(p_clear, 4))
print("P(Congestion AND Clear Weather):", round(p_congestion_and_clear, 4))

