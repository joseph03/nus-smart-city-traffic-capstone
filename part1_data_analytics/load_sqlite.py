import sqlite3
import pandas as pd

csv_file = "data/raw/Metro_Interstate_Traffic_Volume.csv"
db_file = "part1_data_analytics/traffic.db"

# Load CSV
df = pd.read_csv(csv_file)

print("CSV loaded")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Connect to SQLite
conn = sqlite3.connect(db_file)

# Save dataframe as SQLite table
df.to_sql(
    "traffic",
    conn,
    if_exists="replace",
    index=False
)

print("Data loaded into SQLite")

# Verify number of records
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM traffic")
row_count = cursor.fetchone()[0]

print("Rows in SQLite:", row_count)

# Verify table columns
cursor.execute("PRAGMA table_info(traffic)")
columns = cursor.fetchall()

print("\nSQLite columns:")
for column in columns:
    print(column[1], "-", column[2])

conn.close()
