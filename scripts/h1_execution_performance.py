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

def check_normality(df_exec):
    """Check normality of execution times for each function-platform combination"""
    normality_results = []
    
    for (func, platform), group in df_exec.groupby(['name', 'platform']):
        if len(group) > 3:  # Need at least 3 points for normality test
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
    
    # Generate visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='function', y='p_value', hue='platform', style='is_normal',
                    data=normality_df, palette=['#4C72B0', '#DD8452'], s=100)
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
    # Filter out cold starts for execution time analysis
    df_exec = df[df['cold_start'] == False].copy()
    
    # Platform-level analysis
    platform_stats = df_exec.groupby('platform')['execution_ms'].apply(calculate_stats).unstack()
    platform_stats.to_csv(os.path.join(OUTPUT_DIR, "reports", "platform_execution_stats.csv"))
    
    # Function-level analysis
    function_stats = df_exec.groupby(['name', 'platform'])['execution_ms'].apply(calculate_stats).unstack()
    function_stats.to_csv(os.path.join(OUTPUT_DIR, "reports", "function_execution_stats.csv"))
    
    return df_exec, platform_stats, function_stats

def perform_comparative_analysis(df_exec, normality_df=None):
    """Compare Fermyon vs AWS Lambda performance using appropriate statistical tests"""
    functions = df_exec['name'].unique()
    comparison_results = []

    for func in functions:
        fermyon_data = df_exec[(df_exec['name'] == func) & (df_exec['platform'] == 'Fermyon Spin')]['execution_ms']
        aws_data = df_exec[(df_exec['name'] == func) & (df_exec['platform'] == 'AWS Lambda')]['execution_ms']
        
        if len(fermyon_data) > 1 and len(aws_data) > 1:
            # Check normality if results are available
            use_parametric = False
            if normality_df is not None:
                fermyon_normal = normality_df[(normality_df['function'] == func) & 
                                           (normality_df['platform'] == 'Fermyon Spin')]['is_normal'].values[0]
                aws_normal = normality_df[(normality_df['function'] == func) & 
                                        (normality_df['platform'] == 'AWS Lambda')]['is_normal'].values[0]
                use_parametric = fermyon_normal and aws_normal
            
            if use_parametric:
                # Use t-test for normally distributed data
                stat, p_value = stats.ttest_ind(fermyon_data, aws_data, equal_var=False)
                test_used = 't-test'
            else:
                # Use Mann-Whitney U test for non-normal data
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
            'fermyon_faster': fermyon_data.mean() < aws_data.mean()
        })

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "reports", "function_comparison_stats.csv"), index=False)
    return comparison_df

def generate_visualizations(df_exec, comparison_df):
    """Generate all required visualizations with improved styling"""
    # [Previous visualization code remains exactly the same]
    # ... (rest of your existing visualization functions)

def generate_report(comparison_df, normality_df):
    """Generate a summary report with normality information"""
    normal_functions = normality_df[normality_df['is_normal']]['function'].nunique()
    total_functions = normality_df['function'].nunique()
    
    report_summary = f"""
Hypothesis 1 Analysis Report - Handler Execution Performance (execution_ms)

Data Distribution Characteristics:
- {normal_functions} out of {total_functions} function-platform combinations showed normal distribution (Shapiro-Wilk, p > 0.05)
- {comparison_df['test_used'].value_counts().to_string()} tests were used based on normality assessment

Key Findings:
1. For lightweight tasks:
   - Fermyon shows lower mean execution time in {comparison_df[comparison_df['fermyon_faster']].shape[0]} out of {len(comparison_df)} functions
   - Fermyon shows lower CV (more consistent performance) in {sum(comparison_df['fermyon_cv'] < comparison_df['aws_cv'])} out of {len(comparison_df)} functions

2. Statistical Significance:
{comparison_df[['function', 'p_value', 'test_used']].to_string()}

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