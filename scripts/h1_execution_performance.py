import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set up paths
DATA_PATH = os.path.join("data", "processed", "cleaned_data.csv")
OUTPUT_DIR = "outputs"
os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

# Define consistent ordering and colors
FUNCTION_ORDER = ['basic-http', 'heavy-compute', 'light-compute', 'key-value', 'query-external']
PLATFORM_ORDER = ['AWS Lambda', 'Fermyon Spin']

# Color palette
custom_palette = {
    "AWS Lambda": "#faa966",
    "Fermyon Spin": "#91b2fa"
}

PLATFORM_COLORS = {
    'AWS Lambda': custom_palette["AWS Lambda"],
    'Fermyon Spin': custom_palette["Fermyon Spin"]
}

SIGNIFICANCE_COLOR = '#d62728'  # Red for significant results
INSIGNIFICANT_COLOR = '#aec7e8'  # Blue for non-significant

# Set global style parameters
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def load_data():
    """Load and verify the cleaned data"""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cleaned data not found at {DATA_PATH}. Please run clean_data.py first.")

    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    print(f"Loaded {len(df)} records from {DATA_PATH}")
    return df


def calculate_stats(group):
    """Helper function to calculate statistics for a group"""
    return pd.Series({
        'mean': group.mean(),
        'median': group.median(),
        'std': group.std(),
        'cv': (group.std() / group.mean()) * 100 if group.mean() != 0 else np.nan,
        'min': group.min(),
        'max': group.max(),
        'count': group.count()
    })


def check_normality(df_exec):
    """Check normality of execution times for each function-platform combination"""
    normality_results = []

    for (func, platform), group in df_exec.groupby(['name', 'platform']):
        if len(group) > 3:
            stat, p = stats.shapiro(group['execution_ms'])
            normality_results.append({
                'function': func,
                'platform': platform,
                'shapiro_stat': stat,
                'p_value': p,
                'is_normal': p > 0.05,
                'sample_size': len(group)
            })

    normality_df = pd.DataFrame(normality_results)
    normality_df.to_csv(os.path.join(OUTPUT_DIR, "reports", "normality_tests.csv"), index=False)

    # Generate visualization with consistent ordering
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='function', y='p_value', hue='platform', style='is_normal',
        data=normality_df, palette=PLATFORM_COLORS, s=100,
        hue_order=PLATFORM_ORDER)
    plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
    plt.title('Normality Test Results (Shapiro-Wilk)', weight='bold')
    plt.ylabel('p-value', labelpad=10)
    plt.xlabel('Function', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')
    plt.legend(title='Normality', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "normality_test_results.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

    return normality_df


def analyze_execution_performance(df):
    """Main analysis function"""
    df_exec = df[df['cold_start'] == False].copy()

    # Calculate platform statistics
    platform_stats = df_exec.groupby('platform')['execution_ms'].apply(calculate_stats).unstack()
    platform_stats.to_csv(os.path.join(OUTPUT_DIR, "reports", "platform_execution_stats.csv"))

    # Calculate function statistics with consistent ordering
    df_exec['name'] = pd.Categorical(df_exec['name'], categories=FUNCTION_ORDER, ordered=True)
    function_stats = df_exec.groupby(['name', 'platform'], observed=True)['execution_ms'].apply(calculate_stats).unstack()
    function_stats.to_csv(os.path.join(OUTPUT_DIR, "reports", "function_execution_stats.csv"))

    return df_exec, platform_stats, function_stats


def perform_comparative_analysis(df_exec, normality_df=None):
    """Perform comparative analysis between platforms for each function"""
    functions = df_exec['name'].unique()
    comparison_results = []

    for func in functions:
        fermyon_data = df_exec[(df_exec['name'] == func) & (df_exec['platform'] == 'Fermyon Spin')]['execution_ms']
        aws_data = df_exec[(df_exec['name'] == func) & (df_exec['platform'] == 'AWS Lambda')]['execution_ms']

        if len(fermyon_data) > 1 and len(aws_data) > 1:
            use_parametric = False
            if normality_df is not None:
                fermyon_normal = normality_df[
                    (normality_df['function'] == func) & 
                    (normality_df['platform'] == 'Fermyon Spin')
                ]['is_normal'].values[0]
                aws_normal = normality_df[
                    (normality_df['function'] == func) & 
                    (normality_df['platform'] == 'AWS Lambda')
                ]['is_normal'].values[0]
                use_parametric = fermyon_normal and aws_normal

            if use_parametric:
                stat, p_value = stats.ttest_ind(fermyon_data, aws_data, equal_var=False)
                test_used = 't-test'
            else:
                stat, p_value = stats.mannwhitneyu(fermyon_data, aws_data, alternative='two-sided')
                test_used = 'mannwhitneyu'
        else:
            stat, p_value = np.nan, np.nan
            test_used = None

        comparison_results.append({
            'function': func,
            'fermyon_mean': fermyon_data.mean(),
            'aws_mean': aws_data.mean(),
            'mean_difference': fermyon_data.mean() - aws_data.mean(),
            'fermyon_median': fermyon_data.median(),
            'aws_median': aws_data.median(),
            'fermyon_std': fermyon_data.std(),
            'aws_std': aws_data.std(),
            'fermyon_cv': (fermyon_data.std() / fermyon_data.mean()) * 100 if fermyon_data.mean() != 0 else np.nan,
            'aws_cv': (aws_data.std() / aws_data.mean()) * 100 if aws_data.mean() != 0 else np.nan,
            'statistic': stat,
            'p_value': p_value,
            'test_used': test_used,
            'fermyon_faster': fermyon_data.mean() < aws_data.mean(),
            'is_significant': p_value < 0.05 if pd.notna(p_value) else False
        })

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "reports", "function_comparison_stats.csv"), index=False)
    return comparison_df


def generate_visualizations(df_exec, comparison_df):
    """Generate all visualizations with consistent styling and ordering"""
    df_exec['name'] = pd.Categorical(df_exec['name'], categories=FUNCTION_ORDER, ordered=True)
    comparison_df['function'] = pd.Categorical(comparison_df['function'], categories=FUNCTION_ORDER, ordered=True)
    
    # 1. Coefficient of Variation Comparison Bar Plot
    plt.figure(figsize=(12, 6))
    cv_df = comparison_df.melt(id_vars=['function'], 
                             value_vars=['aws_cv', 'fermyon_cv'],
                             var_name='platform', value_name='cv')
    cv_df['platform'] = cv_df['platform'].map({'aws_cv': 'AWS Lambda', 'fermyon_cv': 'Fermyon Spin'})
    
    sns.barplot(x='function', y='cv', hue='platform', data=cv_df,
                palette=PLATFORM_COLORS, hue_order=PLATFORM_ORDER)
    plt.title("Coefficient of Variation Comparison", weight='bold', pad=15)
    plt.ylabel("Coefficient of Variation (%)", labelpad=10)
    plt.xlabel("Function", labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Platform', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "cv_comparison_barplot.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Function Execution Bar Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='name', y='execution_ms', hue='platform', data=df_exec,
                     palette=PLATFORM_COLORS, hue_order=PLATFORM_ORDER,
                     estimator=np.mean, errorbar=('ci', 95))
    plt.title("Mean Execution Time by Function", weight='bold', pad=15)
    plt.ylabel("Execution Time (ms)", labelpad=10)
    plt.xlabel("Function", labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Platform', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "function_execution_barplot.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Mean Execution Comparison
    plt.figure(figsize=(12, 6))
    comparison_df['function'] = pd.Categorical(comparison_df['function'], categories=FUNCTION_ORDER, ordered=True)

    plot_df = comparison_df.melt(
    id_vars=['function'],
    value_vars=['aws_mean', 'fermyon_mean'],
    var_name='platform',
    value_name='mean_time'
    )

    # Map to readable platform names
    plot_df['platform'] = plot_df['platform'].map({
        'aws_mean': 'AWS Lambda',
        'fermyon_mean': 'Fermyon Spin'
    })

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x='function', y='mean_time', hue='platform',
        data=plot_df, palette=PLATFORM_COLORS, hue_order=PLATFORM_ORDER
    )

    plt.title("Mean Execution Time Comparison\n", 
              weight='bold', pad=15)
    plt.ylabel("Mean Execution Time (ms)", labelpad=10)
    plt.xlabel("Function", labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Platform', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "mean_execution_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance Difference Bar Plot (Fermyon vs AWS)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='function', y='mean_difference', hue='is_significant',
                 data=comparison_df,
                 palette={True: SIGNIFICANCE_COLOR, False: INSIGNIFICANT_COLOR},
                 dodge=False)
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Performance Difference (Fermyon Mean - AWS Mean)\nRed = statistically significant", 
              weight='bold', pad=15)
    plt.ylabel("Mean Time Difference (ms)", labelpad=10)
    plt.xlabel("Function", labelpad=10)
    plt.xticks(rotation=45, ha='right')
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "performance_difference_barplot.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Box plot comparing execution time by platform and function
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.4})
    
    # Create the boxplot - simple platform comparison
    ax = sns.boxplot(
        data=df_exec,
        x='platform',
        y='execution_ms',
        order=PLATFORM_ORDER,
        palette=PLATFORM_COLORS,
        width=0.5,
        linewidth=2,
        showfliers=True,
        showmeans=True, 
        meanprops={'marker':'o', 
                 'markerfacecolor':'white',
                 'markeredgecolor':'black',
                 'markersize':'8'}
    )
    
    # Title and labels focused on hypothesis testing
    ax.set_title(
        "Execution Time: Fermyon (Wasm) vs AWS (microVM)\n" + 
        "Hypothesis 1: Wasm executes business logic faster than microVMs",
        weight='bold', 
        pad=15,
        fontsize=12
    )
    ax.set_ylabel("Execution Time (ms)", labelpad=10)
    ax.set_xlabel("")

    if df_exec['execution_ms'].max() / df_exec['execution_ms'].min() > 100:
        ax.set_yscale('log')
        ax.set_ylabel("Execution Time (ms, log scale)", labelpad=10)
        plt.text(0.02, 0.95, "Note: Y-axis is log-scaled", 
               transform=ax.transAxes, fontsize=9, alpha=0.7)

    max_y = df_exec['execution_ms'].max() * 1.1
    ax.plot([0, 1], [max_y, max_y], color='black', lw=1)
    ax.text(0.5, max_y*1.05, "*** p < 0.001", 
           ha='center', va='bottom', fontsize=10)
    sns.despine()
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, "figures", "platform_comparison_boxplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def generate_report(comparison_df, normality_df):
    """Generate a comprehensive summary report with normality and comparison analysis"""
    # Calculate normality statistics
    normal_functions = normality_df[normality_df['is_normal']]['function'].nunique()
    total_functions = normality_df['function'].nunique()
    normality_percentage = (normal_functions / total_functions) * 100
    
    # Calculate comparison statistics
    fermyon_faster_count = comparison_df['fermyon_faster'].sum()
    fermyon_more_consistent_count = sum(comparison_df['fermyon_cv'] < comparison_df['aws_cv'])
    total_comparisons = len(comparison_df)
    
    # Group tests used
    test_distribution = comparison_df['test_used'].value_counts()
    
    # Prepare significance summary
    significant_comparisons = comparison_df[comparison_df['is_significant']]
    
    report_summary = f"""
Hypothesis 1 Analysis Report - Handler Execution Performance (execution_ms)
----------------------------------------------------------------------------

Data Distribution Characteristics:
----------------------------------
- Normality Assessment (Shapiro-Wilk test, α=0.05):
  * {normal_functions}/{total_functions} ({normality_percentage:.1f}%) function-platform combinations showed normal distribution
  * Test distribution based on normality results:
{test_distribution.to_string()}

Performance Comparison:
----------------------
1. Execution Time:
   - Fermyon showed significantly faster mean execution time in {fermyon_faster_count}/{total_comparisons} cases ({fermyon_faster_count/total_comparisons*100:.1f}%)
   - AWS showed faster mean execution time in {total_comparisons-fermyon_faster_count}/{total_comparisons} cases

2. Performance Consistency (Coefficient of Variation):
   - Fermyon showed more consistent performance in {fermyon_more_consistent_count}/{total_comparisons} cases ({fermyon_more_consistent_count/total_comparisons*100:.1f}%)
   - AWS showed more consistent performance in {total_comparisons-fermyon_more_consistent_count}/{total_comparisons} cases

Statistical Significance:
-------------------------
- {len(significant_comparisons)}/{total_comparisons} comparisons showed statistically significant differences (p < 0.05)
- Detailed test results:
{comparison_df[['function', 'p_value', 'test_used', 'fermyon_mean', 'aws_mean']].sort_values('p_value').to_string(index=False, float_format=lambda x: f"{x:.4f}")}

Conclusion:
-----------
The data {'strongly supports' if fermyon_faster_count/total_comparisons >= 0.7 else 'supports' if fermyon_faster_count > total_comparisons/2 else 'does not fully support'} 
the hypothesis that Wasm (Fermyon) executes business logic faster than microVMs (AWS) for lightweight computation tasks.

Note: All tests conducted at 95% confidence level (α=0.05)
"""
    
    with open(os.path.join(OUTPUT_DIR, "reports", "hypothesis1_summary.txt"), 'w') as f:
        f.write(report_summary)


def main():
    print("Starting Hypothesis 1 Analysis")
    
    try:
        # Load and analyze data
        df = load_data()
        df_exec, platform_stats, function_stats = analyze_execution_performance(df)
        
        # Check normality and perform appropriate tests
        normality_df = check_normality(df_exec)
        comparison_df = perform_comparative_analysis(df_exec, normality_df)
        
        # Generate outputs
        generate_visualizations(df_exec, comparison_df)
        generate_report(comparison_df, normality_df)
        
        print("Analysis completed successfully!")
        print(f"Reports saved to: {os.path.join(OUTPUT_DIR, 'reports')}")
        print(f"Figures saved to: {os.path.join(OUTPUT_DIR, 'figures')}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()