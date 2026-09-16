import pandas as pd

df = pd.read_csv("data/processed/traffic_clean.csv")

congestion = df["traffic_volume"] > 5500
clear_weather = df["weather_main"] == "Clear"
cloudy_weather = df["weather_main"] == "Clouds"
high_temp = df["temp"] > 292

# P(Clear Weather | Congestion)
p_clear_given_congestion = (
    (clear_weather & congestion).sum() / congestion.sum()
)

# P(High Temperature | Congestion)
p_high_temp_given_congestion = (
    (high_temp & congestion).sum() / congestion.sum()
)

# Independence check
p_clear = clear_weather.mean()
p_congestion = congestion.mean()
p_clear_and_congestion = (clear_weather & congestion).mean()

independence_product = p_clear * p_congestion

# Odds of congestion in clear weather
clear_congested = (clear_weather & congestion).sum()
clear_not_congested = (clear_weather & ~congestion).sum()
odds_clear = clear_congested / clear_not_congested

# Odds of congestion in cloudy weather
cloudy_congested = (cloudy_weather & congestion).sum()
cloudy_not_congested = (cloudy_weather & ~congestion).sum()
odds_cloudy = cloudy_congested / cloudy_not_congested

odds_ratio = odds_clear / odds_cloudy

print("=== TASK 3.2 CONDITIONAL PROBABILITY ===")
print("P(Clear Weather | Congestion):", round(p_clear_given_congestion, 4))
print("P(High Temperature | Congestion):", round(p_high_temp_given_congestion, 4))

print("\n=== INDEPENDENCE CHECK ===")
print("P(Clear AND Congestion):", round(p_clear_and_congestion, 4))
print("P(Clear) x P(Congestion):", round(independence_product, 4))

print("\n=== ODDS RATIO ===")
print("Odds of congestion in clear weather:", round(odds_clear, 4))
print("Odds of congestion in cloudy weather:", round(odds_cloudy, 4))
print("Odds ratio (Clear / Clouds):", round(odds_ratio, 4))