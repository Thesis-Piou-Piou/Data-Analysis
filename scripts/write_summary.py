import pandas as pd
import os

# Load all summaries
summary = pd.read_csv("outputs/reports/overhead_latency_summary.csv")
penalty = pd.read_csv("outputs/reports/cold_start_penalty_summary.csv")
total_perf = pd.read_csv("outputs/reports/total_performance_summary.csv")

# Prepare output file
report_path = "outputs/reports/analysis_summary.md"
os.makedirs(os.path.dirname(report_path), exist_ok=True)

# Start writing
with open(report_path, "w") as f:
    f.write("# Full Analysis Summary\n\n")

    # Overhead latency section
    f.write("## Overhead Latency Summary (Mean, Median, Std)\n\n")
    f.write("| Platform | Function | Cold Start | Mean (ms) | Median (ms) | Std Dev |\n")
    f.write("|----------|----------|------------|-----------|-------------|---------|\n")
    for _, row in summary.iterrows():
        f.write(f"| {row['platform']} | {row['name']} | {row['cold_start']} | "
                f"{row['mean']:.2f} | {row['median']:.2f} | {row['std']:.2f} |\n")

    f.write("\n## Cold Start Penalty Summary\n\n")
    f.write("| Platform | Function | Cold Mean | Warm Mean | Penalty (%) |\n")
    f.write("|----------|----------|-----------|-----------|--------------|\n")
    for _, row in penalty.iterrows():
        f.write(f"| {row['platform']} | {row['name']} | "
                f"{row['mean_cold']:.2f} | {row['mean_warm']:.2f} | {row['cold_start_penalty_%']:.2f}% |\n")

    # Total performance section
    f.write("\n## Total Performance Summary\n\n")
    f.write("### Mean Total Time (ms)\n\n")
    f.write("| Platform | Function | Cold Start | Mean Total (ms) | Std Dev | Count |\n")
    f.write("|----------|----------|------------|------------------|----------|--------|\n")
    for _, row in total_perf.iterrows():
        f.write(f"| {row['platform']} | {row['name']} | {row['cold_start']} | "
                f"{row['mean_total']:.2f} | {row['std_total']:.2f} | {int(row['count'])} |\n")

    f.write("\n### Execution / Overhead Ratio\n\n")
    f.write("| Platform | Function | Cold Start | Mean Ratio | Std Dev |\n")
    f.write("|----------|----------|------------|------------|----------|\n")
    for _, row in total_perf.iterrows():
        f.write(f"| {row['platform']} | {row['name']} | {row['cold_start']} | "
                f"{row['mean_ratio']:.2f} | {row['std_ratio']:.2f} |\n")

    # Combined key observations
    f.write("\n## Key Observations\n\n")
    all_observations = [
        # Overhead-related
        "Overhead related observations:",
        "AWS Lambda consistently shows significantly higher overhead during cold starts compared to warm starts, "
        "with cold start penalties ranging from 60% to over 170%.",

        "Fermyon Spin demonstrates minimal or even negative cold start penalties in some functions, indicating "
        "extremely low provisioning cost and potential performance stability during initial execution.",

        "Lighter functions like `basic-http`, `key-value`, and `light-compute` suffer disproportionately from AWS cold starts, "
        "leading to poor performance predictability in low-traffic or bursty scenarios.",

        "Fermyon’s overhead measurements are more stable across cold and warm runs, reflected in both lower cold start penalties "
        "and smaller standard deviation values — a strong indicator of consistent platform behavior.",

        "Even for compute-heavy tasks like `heavy-compute`, AWS cold start overhead remains high, suggesting "
        "that cold starts are a persistent bottleneck regardless of workload complexity.",

        "Fermyon appears particularly well-suited for latency-sensitive or on-demand workloads such as microservices, "
        "due to its near-zero cold start impact and predictable overhead latency.",

        "These trends strongly support Hypothesis 2 — that WebAssembly-based platforms like Fermyon Spin significantly reduce "
        "overhead latency and cold start delays compared to traditional serverless platforms.",

        # Total performance-related
        "Total performance related observations:",
        "Fermyon consistently shows lower total time for lightweight functions like `basic-http`, due to much lower overhead.",

        "AWS sometimes outperforms Fermyon in compute-heavy tasks like `heavy-compute`, suggesting better execution scaling.",

        "The execution/overhead ratio is significantly higher on AWS for compute-heavy functions, but lower for I/O-bound ones.",

        "Fermyon maintains a more balanced ratio across functions, reflecting consistent low overhead and good runtime utilization.",

        "Cold starts impact AWS's total time more dramatically than Fermyon's, confirming prior observations from the overhead analysis.",

        "These results support the idea that Fermyon is optimal for latency-sensitive, short-lived workloads, "
        "while AWS may be better suited for heavier compute-bound workloads."
    ]

    for obs in all_observations:
        f.write(f"- {obs}\n")

print(f"Combined analysis report written to: {report_path}")
