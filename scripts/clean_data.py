import pandas as pd
import os

# Paths
INPUT_PATH = "data/raw/cumulative_results.csv"
OUTPUT_PATH = "data/processed/cleaned_data.csv"

# Load data
df = pd.read_csv(INPUT_PATH)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Standardize cold_start to boolean
df["cold_start"] = df["cold_start"].astype(str).str.lower().map({"true": True, "false": False})

# Drop rows with missing critical values
df.dropna(subset=["timestamp", "platform", "name", "execution_ms", "overhead_ms", "total_ms", "cold_start"], inplace=True)

# Report summary
print(f"Loaded {len(df)} cleaned records.")

# Save cleaned file
df.to_csv(OUTPUT_PATH, index=False)
print(f"Cleaned data saved to: {OUTPUT_PATH}")