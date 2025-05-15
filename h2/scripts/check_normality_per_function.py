# To run this script, run this command in the terminal:
# python scripts/check_normality_per_function.py --metrics "overhead_ms" "total_ms"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, probplot
import os
import argparse

def sanitize_filename(text):
    return text.lower().replace(" ", "_").replace("–", "-").replace("|", "").replace("__", "_")

def check_normality_per_function(df, metric, out_csv, fig_dir=None):
    df["platform"] = df["platform"].astype(str).str.strip()
    df["cold_start"] = df["cold_start"].astype(bool)

    functions = df["name"].unique()
    platforms = df["platform"].unique()
    results = []

    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)

    for func in functions:
        for platform in platforms:
            for cold in [True, False]:
                subset = df[
                    (df["name"] == func) &
                    (df["platform"] == platform) &
                    (df["cold_start"] == cold)
                ][metric].dropna()

                label = f"{func} | {platform} | {'Cold' if cold else 'Warm'}"
                print(f" {label} → {len(subset)} samples")

                if len(subset) < 8:
                    print("Too few samples, skipping.\n")
                    continue

                # Shapiro-Wilk test for normality
                p_shapiro = None
                if len(subset) < 5000:
                    _, p_shapiro = shapiro(subset)

                # Summary
                conclusion = f"Shapiro p={p_shapiro:.4f} → {'Not normal' if p_shapiro < 0.05 else 'Possibly normal'}"

                results.append({
                    "function": func,
                    "platform": platform,
                    "cold_start": cold,
                    "n": len(subset),
                    "shapiro_p": round(p_shapiro, 4) if p_shapiro else None,
                    "conclusion": conclusion
                })

                # Save plot
                if fig_dir:
                    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                    sns.histplot(subset, kde=True, ax=axs[0], bins=20)
                    axs[0].set_title(f"Histogram: {label}")
                    probplot(subset, dist="norm", plot=axs[1])
                    axs[1].set_title(f"Q–Q Plot: {label}")
                    plt.tight_layout()

                    fig_path = os.path.join(
                        fig_dir,
                        f"{metric}_{sanitize_filename(label)}.png"
                    )
                    plt.savefig(fig_path)
                    plt.close()
                    print(f"Saved plot: {fig_path}\n")

    # Save results
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\n Normality results saved to: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check normality per function by platform/inferred cold start.")
    parser.add_argument("--file", type=str, default="../data/processed/cleaned_data.csv", help="Input CSV file")
    parser.add_argument("--metrics", type=str, nargs='+', default=["overhead_ms", "total_ms"], help="Columns to test")
    parser.add_argument("--figures", type=str, default="outputs/figures/normality", help="Folder to save plots")

    args = parser.parse_args()
    df = pd.read_csv(args.file, parse_dates=["timestamp"])

    # Loop through each metric and save results to separate files
    for metric in args.metrics:
        out_csv = f"outputs/reports/normality_per_function_{metric}.csv"
        check_normality_per_function(df, metric, out_csv, args.figures)
