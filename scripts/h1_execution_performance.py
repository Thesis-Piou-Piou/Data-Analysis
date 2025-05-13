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

def analyze_execution_performance(df):
    """Main analysis function"""
    # Filter out cold starts for execution time analysis
    df_exec = df[df['cold_start'] == False].copy()
    
    # Platform-level analysis
    platform_stats = df_exec.groupby('platform')['execution_ms'].apply(calculate_stats).unstack()
    platform_stats.to_csv(os.path.join(OUTPUT_DIR, "reports", "platform_execution_stats.csv"))
    
    # Function-level analysis
    function_stats = df_exec.groupby(['name', 'platform'])['execution_ms'].apply(calculate_stats).unstack()
    function_stats.to_csv(os.path.join(OUTPUT_DIR, "reports", "function_execution_stats.csv"))
    
    return df_exec, platform_stats, function_stats

def perform_comparative_analysis(df_exec):
    """Compare Fermyon vs AWS Lambda performance using Mann-Whitney U Test"""
    functions = df_exec['name'].unique()
    comparison_results = []

    for func in functions:
        fermyon_data = df_exec[(df_exec['name'] == func) & (df_exec['platform'] == 'Fermyon Spin')]['execution_ms']
        aws_data = df_exec[(df_exec['name'] == func) & (df_exec['platform'] == 'AWS Lambda')]['execution_ms']
        
        if len(fermyon_data) > 1 and len(aws_data) > 1:
            # Perform Mann-Whitney U Test
            u_stat, p_value = stats.mannwhitneyu(fermyon_data, aws_data, alternative='two-sided')
        else:
            u_stat, p_value = np.nan, np.nan
        
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
            'u_statistic': u_stat,
            'p_value': p_value,
            'fermyon_faster': fermyon_data.mean() < aws_data.mean()
        })

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "reports", "function_comparison_stats.csv"), index=False)
    return comparison_df

def generate_visualizations(df_exec, comparison_df):
    """Generate all required visualizations with improved styling"""
    # Set style and palette
    sns.set_style("white")
    plt.style.use('default')  # Reset to default for clean slate
    custom_palette = ["#4C72B0", "#DD8452"]  # Blue and orange for good contrast
    
    # Set universal style parameters
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.titlepad': 15,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })
    
    # 1. Platform Comparison - Execution Time
    plt.figure(figsize=(10, 6))
    ax1 = sns.boxplot(x='platform', y='execution_ms', data=df_exec, 
                     palette=custom_palette, width=0.5, hue='platform',
                     legend=False)
    plt.title('Execution Time Distribution by Platform', weight='bold')
    plt.ylabel('Execution Time (ms)', labelpad=10)
    plt.xlabel('Platform', labelpad=10)
    plt.yscale('log')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "platform_execution_boxplot.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Function-level Comparison
    plt.figure(figsize=(16, 8))
    ax2 = sns.barplot(
        x='name',
        y='execution_ms',
        hue='platform',
        data=df_exec,
        palette=custom_palette,
        errorbar='sd',
        errwidth=1.5,
        capsize=0.1,
        saturation=0.9
    )
    plt.title('Mean Execution Time by Function and Platform', fontsize=16, weight='bold', pad=12)
    plt.ylabel('Mean Execution Time (ms)', fontsize=13)
    plt.xlabel('Function', fontsize=13)
    plt.yscale('log')

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add value labels for clarity
    for p in ax2.patches:
        height = p.get_height()
        if height > 0 and not np.isnan(height):
            ax2.text(
                p.get_x() + p.get_width() / 2.,
                height * 1.15,  # slightly above for log scale
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=8
            )

    ax2.legend(title='Platform', title_fontsize=12, fontsize=10, frameon=True, loc='upper right')

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "figures", "function_execution_barplot.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    # 3. Coefficient of Variation Comparison
    cv_data = comparison_df.melt(id_vars=['function'], 
                               value_vars=['fermyon_cv', 'aws_cv'],
                               var_name='platform', 
                               value_name='cv')
    cv_data['platform'] = cv_data['platform'].replace({'fermyon_cv': 'Fermyon Spin', 
                                                     'aws_cv': 'AWS Lambda'})

    plt.figure(figsize=(14, 8))
    ax3 = sns.barplot(x='function', y='cv', hue='platform', data=cv_data,
                     palette=custom_palette, saturation=0.85)
    plt.title('Execution Time Consistency (CV%) by Function and Platform', weight='bold')
    plt.ylabel('Coefficient of Variation (%)', labelpad=10)
    plt.xlabel('Function', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    ax3.legend(title='Platform', frameon=True, shadow=True)
    
    # Add value labels
    for p in ax3.patches:
        height = p.get_height()
        if height > 0:
            ax3.text(p.get_x() + p.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "cv_comparison_barplot.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Performance Difference Plot
    plt.figure(figsize=(14, 8))

    # Filter NaNs to avoid plotting errors
    clean_df = comparison_df.dropna(subset=['mean_difference'])

    # Define color mapping based on value
    colors = ["#55A868" if diff < 0 else "#C44E52" for diff in clean_df['mean_difference']]

    ax4 = sns.barplot(
        x='function',
        y='mean_difference',
        data=clean_df,
        palette=colors
    )

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title('Performance Difference: Fermyon vs AWS Lambda', weight='bold')
    plt.ylabel('Mean Time Difference (ms)\n(Fermyon - AWS)', labelpad=10)
    plt.xlabel('Function', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    ax4.grid(axis='y', linestyle='--', alpha=0.3)

    # Add value labels to bars
    for bar in ax4.patches:
        height = bar.get_height()
        if not np.isnan(height):
            label_y = height + 2 if height > 0 else height - 6
            ax4.text(
                bar.get_x() + bar.get_width() / 2.,
                label_y,
                f'{height:.1f}',
                ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "figures", "performance_difference_barplot.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

def generate_report(comparison_df):
    """Generate a summary report"""
    report_summary = f"""
Hypothesis 1 Analysis Report - Handler Execution Performance (execution_ms)

Key Findings:
1. For lightweight tasks:
   - Fermyon shows lower mean execution time in {comparison_df[comparison_df['fermyon_faster']].shape[0]} out of {len(comparison_df)} functions
   - Fermyon shows lower CV (more consistent performance) in {sum(comparison_df['fermyon_cv'] < comparison_df['aws_cv'])} out of {len(comparison_df)} functions

2. Statistical Significance:
{comparison_df[['function', 'p_value']].to_string()}

Conclusion:
The data {'supports' if sum(comparison_df['fermyon_faster']) > sum(~comparison_df['fermyon_faster']) else 'does not fully support'} the hypothesis that Wasm (Fermyon) executes business logic faster than microVMs (AWS) for lightweight computation tasks.
"""
    with open(os.path.join(OUTPUT_DIR, "reports", "hypothesis1_summary.txt"), 'w') as f:
        f.write(report_summary)

def main():
    print("Starting Hypothesis 1 Analysis...")
    
    try:
        # Load and analyze data
        df = load_data()
        df_exec, platform_stats, function_stats = analyze_execution_performance(df)
        comparison_df = perform_comparative_analysis(df_exec)
        
        # Generate outputs
        generate_visualizations(df_exec, comparison_df)
        generate_report(comparison_df)
        
        print("Analysis completed successfully!")
        print(f"Reports saved to: {os.path.join(OUTPUT_DIR, 'reports')}")
        print(f"Figures saved to: {os.path.join(OUTPUT_DIR, 'figures')}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()