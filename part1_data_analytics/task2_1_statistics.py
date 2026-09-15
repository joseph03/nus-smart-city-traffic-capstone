import pandas as pd

csv_file = "data/processed/traffic_clean.csv"

df = pd.read_csv(csv_file)

traffic = df["traffic_volume"]

print("=== TRAFFIC VOLUME STATISTICS ===")

mean_value = traffic.mean()
median_value = traffic.median()
std_value = traffic.std()
variance_value = traffic.var()
range_value = traffic.max() - traffic.min()

print("Mean:", round(mean_value, 2))
print("Median:", round(median_value, 2))
print("Standard deviation:", round(std_value, 2))
print("Variance:", round(variance_value, 2))
print("Range:", range_value)

