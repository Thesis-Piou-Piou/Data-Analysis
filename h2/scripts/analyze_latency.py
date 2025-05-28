# Use this command to run this script in the terminal: python scripts/analyze_latency.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ── Visualization 1: Grouped bar chart ──

# Load precomputed summary for overhead latency
overhead_summary = pd.read_csv("outputs/reports/overhead_latency_summary.csv")

# Add 'platform_cold' column to summary for plotting
overhead_summary["platform_cold"] = overhead_summary.apply(
    lambda row: f"{row['platform']} – {'Inferred Cold Start' if row['cold_start'] else 'Warm Start'}", axis=1
)

# Custom color palette
custom_palette = {
    "AWS Lambda – Inferred Cold Start": "#faddc5",
    "AWS Lambda – Warm Start": "#faa966",
    "Fermyon Spin – Inferred Cold Start": "#dfe7fa",
    "Fermyon Spin – Warm Start": "#91b2fa"
}

plt.figure(figsize=(14, 7))
ax1 = sns.barplot(
    data=overhead_summary,
    x="name",
    y="mean",
    hue="platform_cold",
    palette=custom_palette,
    hue_order=["AWS Lambda – Inferred Cold Start", "AWS Lambda – Warm Start", "Fermyon Spin – Inferred Cold Start", "Fermyon Spin – Warm Start"],
    ci=None,
    edgecolor=None,
    dodge=True,
    zorder=3,
    width=0.8
)

# Add value labels on top of the bars with proper formatting
for i, bar in enumerate(ax1.patches):
    height = bar.get_height()
    if height > 0:  # Only add label for bars with height greater than zero
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.2f}",  # Format with two decimal places
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax1.set_ylim(0,overhead_summary["mean"].max() * 1.1) # Set y-axis limit to better visualize values

ax1.set_ylabel("Mean Overhead (ms)")
ax1.set_xlabel("Function")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

ax1.legend(
    title="Platform / Start Type",
    loc='upper left',  # Location of the legend
    bbox_to_anchor=(1.05, 1),  # Position the legend outside the plot
    borderaxespad=0.,  # Padding between legend and plot
    fontsize=10
)

ax1.grid(True, axis='y', linestyle='-', linewidth=1, zorder=1 )

plt.tight_layout()
plt.savefig("outputs/figures/mean_overhead_platform_coldwarm.png")
plt.close()
print("Saved: mean_overhead_platform_coldwarm.png")

# ── Visualization 2: Cold start penalty plot ──

# Load precomputed cold start penalty summary
penalty = pd.read_csv("outputs/reports/cold_start_penalty_summary.csv")

# Ensure the platform column is consistent for plotting
penalty["platform"] = penalty["platform"].replace({
    "AWS Lambda": "AWS Lambda",
    "Fermyon Spin": "Fermyon Spin"
})

# Custom color palette for AWS and Fermyon
custom_palette_penalty = {
    "AWS Lambda": "#f99a4b",  # orange for AWS Lambda
    "Fermyon Spin": "#76a0fa"  # blue for Fermyon Spin
}

plt.figure(figsize=(12, 6))
ax2 = sns.barplot(
    data=penalty,
    x="name",
    y="cold_start_penalty_%",
    hue="platform",
    palette=custom_palette_penalty,
    errorbar=None,
    dodge=True,
    zorder=3
)

# Add value labels on top of the bars
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

ax2.set_ylabel("Inferred Cold Start Penalty (%)")
ax2.set_xlabel("Function")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.legend(
    title="Platform",
    loc='upper left',
    bbox_to_anchor=(1.05, 1),
    borderaxespad=0.,
    fontsize=10
)
ax2.grid(True, axis='y', linestyle='-', linewidth=1, zorder=1 )

# Manually set y-axis limits to add padding between the highest/lowest values and the plot margins
padding = 5
ymin, ymax = ax2.get_ylim()
ax2.set_ylim(ymin - padding, ymax + padding)

plt.tight_layout()
plt.savefig("outputs/figures/cold_start_penalty_labeled.png")
plt.close()
print("Saved: cold_start_penalty_labeled.png")

# ── Visualization 3: Dot Plot for Overhead Latency ──
plt.figure(figsize=(14, 7))

df = pd.read_csv("../data/processed/cleaned_data.csv", parse_dates=["timestamp"])
df["cold_start"] = df["cold_start"].astype(bool)

# Update platform_cold label for consistency with bar chart
df["platform_cold"] = df.apply(
    lambda row: f"{row['platform']} – {'Inferred Cold Start' if row['cold_start'] else 'Warm Start'}", axis=1
)

# Custom color palette for categories (AWS Lambda vs Fermyon Spin)
custom_palette_dotplot = {
    "AWS Lambda – Inferred Cold Start": "#faddc5",
    "AWS Lambda – Warm Start": "#faa966",
    "Fermyon Spin – Inferred Cold Start": "#dfe7fa",
    "Fermyon Spin – Warm Start": "#91b2fa"
}

hue_order = [
    "AWS Lambda – Inferred Cold Start",
    "AWS Lambda – Warm Start",
    "Fermyon Spin – Inferred Cold Start",
    "Fermyon Spin – Warm Start"
]

# Sort the function names alphabetically
sorted_function_names = sorted(df['name'].unique())

# Create the strip plot (dot plot) with jitter to separate points
ax3 = sns.stripplot(
    data=df,
    x="overhead_ms",
    y="name",
    hue="platform_cold",
    palette=custom_palette_dotplot,
    jitter=True,  # Add jitter to separate points along the x-axis
    dodge=True,  # Ensure points from different categories (e.g., cold vs warm) are placed separately
    marker="o",  # Use circles for the points
    edgecolor="gray",
    order=sorted_function_names,
    hue_order=hue_order 
)

# Set plot labels
ax3.set_xlabel("Overhead Latency (ms)")
ax3.set_ylabel("Function")
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)

ax3.legend(
    title="Platform / Start Type",
    loc='upper left',
    bbox_to_anchor=(1.05, 1),
    borderaxespad=0.,
    fontsize=10
)

# Tight layout to prevent overlapping
plt.tight_layout()

# Save the plot
plt.savefig("outputs/figures/overhead_latency_dot_plot.png")
plt.close()
print("Saved: overhead_latency_dot_plot.png")

# ── Visualization 4: Dot plot for Total Execution Time ──

df = pd.read_csv("../data/processed/cleaned_data.csv", parse_dates=["timestamp"])
df["cold_start"] = df["cold_start"].astype(bool)
df["platform_cold"] = df.apply(
    lambda row: f"{row['platform']} – {'Inferred Cold Start' if row['cold_start'] else 'Warm Start'}", axis=1
)

# Custom color palette for categories (AWS Lambda vs Fermyon Spin)
custom_palette_dotplot = {
    "AWS Lambda – Inferred Cold Start": "#faddc5",
    "AWS Lambda – Warm Start": "#faa966",
    "Fermyon Spin – Inferred Cold Start": "#dfe7fa",
    "Fermyon Spin – Warm Start": "#91b2fa"
}

hue_order = [
    "AWS Lambda – Inferred Cold Start",
    "AWS Lambda – Warm Start",
    "Fermyon Spin – Inferred Cold Start",
    "Fermyon Spin – Warm Start"
]

# Sort the function names alphabetically
sorted_function_names_total = sorted(df['name'].unique())

# Create the strip plot (dot plot) with jitter to separate points
plt.figure(figsize=(14, 7))
ax4 = sns.stripplot(
    data=df,
    x="total_ms",
    y="name",
    hue="platform_cold",
    palette=custom_palette_dotplot,
    jitter=True,
    dodge=True,
    marker="o",
    edgecolor='gray',
    order=sorted_function_names_total,
    hue_order=hue_order 
)

# Set plot labels
ax4.set_xlabel("Total Execution Time (ms)")
ax4.set_ylabel("Function")
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)

ax4.legend(
    title="Platform / Start Type",
    loc='upper left',
    bbox_to_anchor=(1.05, 1),
    borderaxespad=0.,
    fontsize=10
)

# Tight layout to prevent overlapping
plt.tight_layout()

# Save the plot
plt.savefig("outputs/figures/total_execution_time_dot_plot.png")
plt.close()
print("Saved: total_execution_time_dot_plot.png")

# ── Visualization 5 - Boxplot: Overhead Latency Distribution by Function and Cold/Warm Start ──
plt.figure(figsize=(14, 7))

# Sort the data by 'name' to ensure alphabetical order
df["platform_cold"] = df.apply(
    lambda row: f"{row['platform']} – {'Inferred Cold Start' if row['cold_start'] else 'Warm Start'}", axis=1
)

# Sort the 'name' column alphabetically
sorted_function_names = sorted(df['name'].unique())

# Custom color palette for AWS Lambda and Fermyon Spin
custom_palette_boxplot = {
    "AWS Lambda – Inferred Cold Start": "#faddc5",
    "AWS Lambda – Warm Start": "#faa966", 
    "Fermyon Spin – Inferred Cold Start": "#dfe7fa",  
    "Fermyon Spin – Warm Start": "#91b2fa"
}

hue_order = [
    "AWS Lambda – Inferred Cold Start",
    "AWS Lambda – Warm Start",
    "Fermyon Spin – Inferred Cold Start",
    "Fermyon Spin – Warm Start"
]

# Create boxplot
ax = sns.boxplot(
    data=df,
    x="name",
    y="overhead_ms",
    hue="platform_cold",
    palette=custom_palette_boxplot,
    order=sorted_function_names,
    hue_order=hue_order,
    #flierprops=dict(marker="o", color="white", markersize=0)  # Hide outliers
)

# Set plot labels
ax.set_ylabel("Overhead Latency (ms)")
ax.set_xlabel("Function")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Custom legend labels for clarity
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["AWS Lambda – Inferred Cold Start", "AWS Lambda – Warm Start", "Fermyon Spin – Inferred Cold Start", "Fermyon Spin – Warm Start"], title="Platform / Start Type")

plt.tight_layout()
plt.savefig("outputs/figures/boxplot_overhead_latency_by_function.png")
plt.close()
print("Saved: outputs/figures/boxplot_overhead_latency_by_function.png")

# ── Visualization 6: Total Execution Time by Function and Platform ──

# Load total performance summary csv file
total_exec_df = pd.read_csv("outputs/reports/total_performance_summary.csv")
total_exec_df["platform_cold"] = total_exec_df.apply(
    lambda row: f"{row['platform']} – {'Inferred Cold Start' if row['cold_start'] else 'Warm Start'}", axis=1
)

# Custom color palette
custom_palette_exec = {
    "AWS Lambda – Inferred Cold Start": "#faddc5",
    "AWS Lambda – Warm Start": "#faa966",
    "Fermyon Spin – Inferred Cold Start": "#dfe7fa",
    "Fermyon Spin – Warm Start": "#91b2fa"
}

hue_order = [
    "AWS Lambda – Inferred Cold Start",
    "AWS Lambda – Warm Start",
    "Fermyon Spin – Inferred Cold Start",
    "Fermyon Spin – Warm Start"
]

plt.figure(figsize=(14, 7))
ax6 = sns.barplot(
    data=total_exec_df,
    x="name",
    y="mean_total",
    hue="platform_cold",
    palette=custom_palette_exec,
    hue_order=hue_order,
    ci=None,
    edgecolor=None,
    dodge=True,
    zorder=3,
    width=0.8
)

# Add value labels
for bar in ax6.patches:
    height = bar.get_height()
    if height > 0:
        ax6.text(
            bar.get_x() + bar.get_width() / 2,
            height + 5,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=6
        )

ax6.set_ylabel("Mean Total Execution Time (ms)")
ax6.set_xlabel("Function")
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
ax6.legend(
    title="Platform / Start Type",
    loc='upper left',
    bbox_to_anchor=(1.05, 1),
    borderaxespad=0.,
    fontsize=10
)
ax6.grid(True, axis='y', linestyle='-', linewidth=1, zorder=1)

plt.tight_layout()
plt.savefig("outputs/figures/mean_total_execution_time_by_platform.png")
plt.close()
print("Saved: mean_total_execution_time_by_platform.png")
