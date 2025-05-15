# python scripts/run_statistical_test.py --metrics "overhead_ms" "total_ms"

import pandas as pd
from scipy.stats import mannwhitneyu
import argparse
import os

def test_per_function(df, metric="overhead_ms"):
    df["platform"] = df["platform"].str.strip()
    df["cold_start"] = df["cold_start"].astype(bool)
    functions = df["name"].unique()
    results = []

    def run_test(group1, group2, label1, label2, context):
        try:
            stat, p = mannwhitneyu(group1, group2, alternative="two-sided")
            return {
                "comparison": f"{label1} vs {label2}",
                "context": context,
                "n_1": len(group1),
                "n_2": len(group2),
                "statistic": round(stat, 4),
                "p_value": round(p, 4),
                "significant": "Yes" if p < 0.05 else "No"
            }
        except ValueError:
            return {
                "comparison": f"{label1} vs {label2}",
                "context": context,
                "n_1": len(group1),
                "n_2": len(group2),
                "statistic": None,
                "p_value": None,
                "significant": "Insufficient data"
            }

    for func in functions:
        subset = df[df["name"] == func]

        # Extract subgroups based on platforms and cold/warm starts
        aws_cold = subset[(subset["platform"] == "AWS Lambda") & (subset["cold_start"])][metric].dropna()
        aws_warm = subset[(subset["platform"] == "AWS Lambda") & (~subset["cold_start"])][metric].dropna()
        fermyon_cold = subset[(subset["platform"] == "Fermyon Spin") & (subset["cold_start"])][metric].dropna()
        fermyon_warm = subset[(subset["platform"] == "Fermyon Spin") & (~subset["cold_start"])][metric].dropna()

        # Run tests for each comparison
        results.append(run_test(aws_cold, fermyon_cold, "AWS Cold", "Fermyon Cold", "cross-platform cold"))
        results.append(run_test(aws_warm, fermyon_warm, "AWS Warm", "Fermyon Warm", "cross-platform warm"))
        results.append(run_test(aws_cold, aws_warm, "AWS Cold", "AWS Warm", "within-AWS"))
        results.append(run_test(fermyon_cold, fermyon_warm, "Fermyon Cold", "Fermyon Warm", "within-Fermyon"))

    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run per-function Mann–Whitney U tests for latency metrics.")
    parser.add_argument("--file", type=str, default="../data/processed/cleaned_data.csv", help="Path to cleaned CSV file")
    parser.add_argument("--metrics", type=str, nargs='+', default=["overhead_ms", "total_ms"], help="Metrics to test (e.g. overhead_ms, total_ms)")
    
    args = parser.parse_args()
    df = pd.read_csv(args.file, parse_dates=["timestamp"])

    # Loop through each metric and save results to separate files
    for metric in args.metrics:
        # Generate output file name based on metric
        out_csv = f"outputs/reports/per_function_{metric}_mannwhitney_tests.csv"
        
        # Run tests for the current metric(s)
        results = test_per_function(df, metric=metric)

        # Ensure the output folder exists
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        
        # Save the results to the appropriate CSV file
        results.to_csv(out_csv, index=False)
        print(f"Per-function Mann-Whitney U test results saved to: {out_csv}")
