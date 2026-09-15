import pandas as pd

csv_file = "data/processed/traffic_clean.csv"

df = pd.read_csv(csv_file)

correlation = df["temp"].corr(df["traffic_volume"])

print("=== TEMPERATURE / TRAFFIC CORRELATION ===")
print("Correlation coefficient:", round(correlation, 4))