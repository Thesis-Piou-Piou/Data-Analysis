import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned data
df = pd.read_csv("data/processed/cleaned_data.csv", parse_dates=["timestamp"])
df["cold_start"] = df["cold_start"].astype(bool)

# Group by platform, function name, and cold_start
summary = (
    df.groupby(["platform", "name", "cold_start"])["overhead_ms"]
    .agg(mean="mean", median="median", std="std", count="count")
    .reset_index()
)

# Save summary
summary.to_csv("outputs/reports/overhead_latency_summary.csv", index=False)
print("✅ Saved: overhead_latency_summary.csv")

# Cold start penalty calculation
cold = summary[summary["cold_start"] == True].copy()
warm = summary[summary["cold_start"] == False].copy()

penalty = pd.merge(cold, warm, on=["platform", "name"], suffixes=("_cold", "_warm"))
penalty["cold_start_penalty_%"] = (
    (penalty["mean_cold"] - penalty["mean_warm"]) / penalty["mean_warm"] * 100
).round(2)

penalty.to_csv("outputs/reports/cold_start_penalty_summary.csv", index=False)
print("Saved: cold_start_penalty_summary.csv")

# ── Visualization 1: Grouped bar chart with SD error bars & labels ──
summary["platform_cold"] = summary.apply(
    lambda row: f"{row['platform']} – {'Cold' if row['cold_start'] else 'Warm'}", axis=1
)

hue_order = [
    "AWS Lambda – Cold",
    "AWS Lambda – Warm",
    "Fermyon Spin – Cold",
    "Fermyon Spin – Warm"
]

custom_palette = {
    "AWS Lambda – Cold": "#1f77b4",    # blue
    "AWS Lambda – Warm": "#ffdf00",    # yellow
    "Fermyon Spin – Cold": "#9467bd",  # purple
    "Fermyon Spin – Warm": "#ff7f0e"   # orange
}

plt.figure(figsize=(14, 7))
ax1 = sns.barplot(
    data=summary,
    x="name",
    y="mean",
    hue="platform_cold",
    palette=custom_palette,
    hue_order=hue_order,
    ci=None,
    edgecolor=".2"
)

# Add manual error bars
for i, bar in enumerate(ax1.patches):
    # Get the row from summary that matches this bar
    if i >= len(summary):
        break  # safety guard
    row = summary.iloc[i]
    height = bar.get_height()
    std = row["std"]
    
    # Error bar
    ax1.errorbar(
        bar.get_x() + bar.get_width() / 2,
        height,
        yerr=std,
        fmt='none',
        ecolor='black',
        capsize=3,
        linewidth=1
    )
    
    # Add value label
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        height + std + 2,
        f"{height:.1f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

ax1.set_title("Mean Overhead Latency by Function (Cold vs Warm, by Platform)")
ax1.set_ylabel("Mean Overhead (ms)")
ax1.set_xlabel("Function")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.legend(title="Platform / Start Type")
plt.tight_layout()
plt.savefig("outputs/figures/mean_overhead_platform_coldwarm_sd.png")
plt.close()
print("Updated: mean_overhead_platform_coldwarm_sd.png with manual SD error bars")

# ── Visualization 2: Cold start penalty plot with labels ──
penalty["platform"] = penalty["platform"].replace({
    "AWS Lambda": "AWS",
    "Fermyon Spin": "Fermyon"
})

plt.figure(figsize=(12, 6))
ax2 = sns.barplot(
    data=penalty,
    x="name",
    y="cold_start_penalty_%",
    hue="platform",
    palette="Set1",
    errorbar=None
)
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)

for container in ax2.containers:
    for bar in container:
        height = bar.get_height()
        offset = 2 if height >= 0 else -5
        va = "bottom" if height >= 0 else "top"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:.1f}%",
            ha="center",
            va=va,
            fontsize=9
        )

ax2.set_title("Cold Start Penalty (%) by Function and Platform")
ax2.set_ylabel("Cold Start Penalty (%)")
ax2.set_xlabel("Function")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.legend(title="Platform")
plt.tight_layout()
plt.savefig("outputs/figures/cold_start_penalty_labeled.png")
plt.close()
print("Saved: cold_start_penalty_labeled.png")

# ── Visualisation 3 - Boxplot: Overhead Latency Distribution by Function and Cold/Warm Start ──
plt.figure(figsize=(14, 7))
df["platform_cold"] = df.apply(
    lambda row: f"{row['platform']} – {'Cold' if row['cold_start'] else 'Warm'}", axis=1
)

ax = sns.boxplot(
    data=df,
    x="name",
    y="overhead_ms",
    hue="platform_cold",
    palette="Set2"
)

ax.set_title("Overhead Latency Distribution by Function (Boxplot)")
ax.set_ylabel("Overhead Latency (ms)")
ax.set_xlabel("Function")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.legend(title="Platform / Start Type")

plt.tight_layout()
plt.savefig("outputs/figures/boxplot_overhead_latency_by_function.png")
plt.close()
print("Saved: outputs/figures/boxplot_overhead_latency_by_function.png")

# Part 2: Analyze total execution time/overhead

# Calculate execution/overhead ratio
df["exec_overhead_ratio"] = df["execution_ms"] / df["overhead_ms"]

# Group by platform, function, and cold_start
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

# Save to CSV
total_perf_path = "outputs/reports/total_performance_summary.csv"
total_perf.to_csv(total_perf_path, index=False)
print(f"Saved: {total_perf_path}")

# ── Visualization 4: Mean Total Time with SD Error Bars & Labels ──
total_perf["platform_cold"] = total_perf.apply(
    lambda row: f"{row['platform']} – {'Cold' if row['cold_start'] else 'Warm'}", axis=1
)

hue_order = [
    "AWS Lambda – Cold",
    "AWS Lambda – Warm",
    "Fermyon Spin – Cold",
    "Fermyon Spin – Warm"
]

custom_palette = {
    "AWS Lambda – Cold": "#1f77b4",    # blue
    "AWS Lambda – Warm": "#ffdf00",    # yellow
    "Fermyon Spin – Cold": "#9467bd",  # purple
    "Fermyon Spin – Warm": "#ff7f0e"   # orange
}

plt.figure(figsize=(14, 7))
ax = sns.barplot(
    data=total_perf,
    x="name",
    y="mean_total",
    hue="platform_cold",
    palette=custom_palette,
    hue_order=hue_order,
    ci=None,
    edgecolor=".2"
)

# Add manual error bars and value labels
for i, bar in enumerate(ax.patches):
    if i >= len(total_perf):
        break  # safety guard

    row = total_perf.iloc[i]
    height = bar.get_height()
    std = row["std_total"]

    # Error bar
    ax.errorbar(
        bar.get_x() + bar.get_width() / 2,
        height,
        yerr=std,
        fmt='none',
        ecolor='black',
        capsize=3,
        linewidth=1
    )

    # Value label
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + std + 2,
        f"{height:.1f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

ax.set_title("Mean Total Time by Function (Cold vs Warm, by Platform)")
ax.set_ylabel("Mean Total Time (ms)")
ax.set_xlabel("Function")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.legend(title="Platform / Start Type")

plt.tight_layout()
plt.savefig("outputs/figures/mean_total_time_by_function.png")
plt.close()
print("Saved: outputs/figures/mean_total_time_by_function.png")

# Vsualisation 5 - Bar plot: Execution / Overhead Ratio
plt.figure(figsize=(14, 7))
ax = sns.barplot(
    data=total_perf,
    x="name",
    y="mean_ratio",
    hue="platform_cold",
    hue_order=hue_order,
    palette=custom_palette,
    ci=None
)

# Add value labels
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        xpos = bar.get_x() + bar.get_width() / 2
        ax.text(
            xpos,
            height + 0.1,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

ax.set_title("Execution / Overhead Ratio by Function (Cold vs Warm, by Platform)")
ax.set_ylabel("Execution / Overhead Ratio")
ax.set_xlabel("Function")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.legend(title="Platform / Start Type")

plt.tight_layout()
plt.savefig("outputs/figures/exec_overhead_ratio_by_function.png")
plt.close()
print("Saved: outputs/figures/exec_overhead_ratio_by_function.png")
