# Use this command to run this script in the terminal: python scripts/generate_csv_summaries.py
import pandas as pd
import os

df = pd.read_csv("../data/processed/cleaned_data.csv", parse_dates=["timestamp"])
df["cold_start"] = df["cold_start"].astype(bool)

# ── Overhead latency summary ──
summary = (
    df.groupby(["platform", "name", "cold_start"])["overhead_ms"]
    .agg(mean="mean", median="median", std="std", count="count")
    .reset_index()
)
os.makedirs("outputs/reports", exist_ok=True)
summary.to_csv("outputs/reports/overhead_latency_summary.csv", index=False)
print("Saved: overhead_latency_summary.csv")

# ── Cold start penalty summary ──
cold = summary[summary["cold_start"] == True].copy()
warm = summary[summary["cold_start"] == False].copy()
penalty = pd.merge(cold, warm, on=["platform", "name"], suffixes=("_cold", "_warm"))
penalty["cold_start_penalty_%"] = (
    (penalty["mean_cold"] - penalty["mean_warm"]) / penalty["mean_warm"] * 100
).round(2)
penalty.to_csv("outputs/reports/cold_start_penalty_summary.csv", index=False)
print("Saved: cold_start_penalty_summary.csv")

# ── Total performance summary ──
df["exec_overhead_ratio"] = df["execution_ms"] / df["overhead_ms"]
total_perf = (
    df.groupby(["platform", "name", "cold_start"])
    .agg(
        mean_total=("total_ms", "mean"),
        std_total=("total_ms", "std"),
        mean_ratio=("exec_overhead_ratio", "mean"),
        std_ratio=("exec_overhead_ratio", "std"),
        count=("total_ms", "count")
    )
    .reset_index()
)
total_perf.to_csv("outputs/reports/total_performance_summary.csv", index=False)
print("Saved: total_performance_summary.csv")
