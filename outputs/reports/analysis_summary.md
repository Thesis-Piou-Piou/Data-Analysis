# Full Analysis Summary

## Overhead Latency Summary (Mean, Median, Std)

| Platform | Function | Cold Start | Mean (ms) | Median (ms) | Std Dev |
|----------|----------|------------|-----------|-------------|---------|
| AWS Lambda | basic-http | False | 470.74 | 459.88 | 77.56 |
| AWS Lambda | basic-http | True | 805.00 | 796.63 | 85.64 |
| AWS Lambda | heavy-compute | False | 467.20 | 459.61 | 76.51 |
| AWS Lambda | heavy-compute | True | 811.48 | 793.11 | 85.16 |
| AWS Lambda | key-value | False | 415.33 | 399.20 | 80.49 |
| AWS Lambda | key-value | True | 1156.21 | 1115.83 | 208.87 |
| AWS Lambda | light-compute | False | 467.29 | 453.32 | 91.27 |
| AWS Lambda | light-compute | True | 779.17 | 766.25 | 87.79 |
| AWS Lambda | query-external | False | 414.82 | 405.44 | 80.54 |
| AWS Lambda | query-external | True | 878.80 | 836.42 | 186.45 |
| Fermyon Spin | basic-http | False | 154.36 | 135.37 | 76.05 |
| Fermyon Spin | basic-http | True | 248.70 | 240.67 | 77.09 |
| Fermyon Spin | heavy-compute | False | 149.03 | 129.19 | 85.99 |
| Fermyon Spin | heavy-compute | True | 137.68 | 106.80 | 75.17 |
| Fermyon Spin | key-value | False | 143.58 | 125.57 | 81.25 |
| Fermyon Spin | key-value | True | 135.63 | 100.19 | 74.09 |
| Fermyon Spin | light-compute | False | 143.44 | 129.23 | 74.17 |
| Fermyon Spin | light-compute | True | 143.06 | 119.61 | 72.15 |
| Fermyon Spin | query-external | False | 147.45 | 125.07 | 86.09 |
| Fermyon Spin | query-external | True | 139.10 | 110.51 | 76.41 |

## Cold Start Penalty Summary

| Platform | Function | Cold Mean | Warm Mean | Penalty (%) |
|----------|----------|-----------|-----------|--------------|
| AWS Lambda | basic-http | 805.00 | 470.74 | 71.01% |
| AWS Lambda | heavy-compute | 811.48 | 467.20 | 73.69% |
| AWS Lambda | key-value | 1156.21 | 415.33 | 178.38% |
| AWS Lambda | light-compute | 779.17 | 467.29 | 66.74% |
| AWS Lambda | query-external | 878.80 | 414.82 | 111.85% |
| Fermyon Spin | basic-http | 248.70 | 154.36 | 61.12% |
| Fermyon Spin | heavy-compute | 137.68 | 149.03 | -7.61% |
| Fermyon Spin | key-value | 135.63 | 143.58 | -5.54% |
| Fermyon Spin | light-compute | 143.06 | 143.44 | -0.27% |
| Fermyon Spin | query-external | 139.10 | 147.45 | -5.66% |

## Total Performance Summary

### Mean Total Time (ms)

| Platform | Function | Cold Start | Mean Total (ms) | Std Dev | Count |
|----------|----------|------------|------------------|----------|--------|
| AWS Lambda | basic-http | False | 470.78 | 77.56 | 120 |
| AWS Lambda | basic-http | True | 805.02 | 85.64 | 30 |
| AWS Lambda | heavy-compute | False | 862.62 | 78.10 | 120 |
| AWS Lambda | heavy-compute | True | 1654.85 | 124.22 | 30 |
| AWS Lambda | key-value | False | 415.33 | 80.49 | 120 |
| AWS Lambda | key-value | True | 1156.21 | 208.87 | 30 |
| AWS Lambda | light-compute | False | 467.32 | 91.27 | 120 |
| AWS Lambda | light-compute | True | 779.58 | 88.06 | 30 |
| AWS Lambda | query-external | False | 444.05 | 82.88 | 120 |
| AWS Lambda | query-external | True | 1508.76 | 188.29 | 30 |
| Fermyon Spin | basic-http | False | 154.37 | 76.05 | 120 |
| Fermyon Spin | basic-http | True | 248.72 | 77.09 | 30 |
| Fermyon Spin | heavy-compute | False | 5560.01 | 119.89 | 120 |
| Fermyon Spin | heavy-compute | True | 5567.85 | 124.71 | 30 |
| Fermyon Spin | key-value | False | 152.72 | 80.88 | 120 |
| Fermyon Spin | key-value | True | 149.64 | 72.88 | 30 |
| Fermyon Spin | light-compute | False | 143.78 | 74.17 | 120 |
| Fermyon Spin | light-compute | True | 143.39 | 72.15 | 30 |
| Fermyon Spin | query-external | False | 680.14 | 93.04 | 120 |
| Fermyon Spin | query-external | True | 681.46 | 79.37 | 30 |

### Execution / Overhead Ratio

| Platform | Function | Cold Start | Mean Ratio | Std Dev |
|----------|----------|------------|------------|----------|
| AWS Lambda | basic-http | False | 0.00 | 0.00 |
| AWS Lambda | basic-http | True | 0.00 | 0.00 |
| AWS Lambda | heavy-compute | False | 0.87 | 0.15 |
| AWS Lambda | heavy-compute | True | 1.05 | 0.15 |
| AWS Lambda | key-value | False | 0.00 | 0.00 |
| AWS Lambda | key-value | True | 0.00 | 0.00 |
| AWS Lambda | light-compute | False | 0.00 | 0.00 |
| AWS Lambda | light-compute | True | 0.00 | 0.00 |
| AWS Lambda | query-external | False | 0.07 | 0.02 |
| AWS Lambda | query-external | True | 0.74 | 0.12 |
| Fermyon Spin | basic-http | False | 0.00 | 0.00 |
| Fermyon Spin | basic-http | True | 0.00 | 0.00 |
| Fermyon Spin | heavy-compute | False | 49.13 | 25.10 |
| Fermyon Spin | heavy-compute | True | 52.07 | 25.78 |
| Fermyon Spin | key-value | False | 0.09 | 0.07 |
| Fermyon Spin | key-value | True | 0.14 | 0.12 |
| Fermyon Spin | light-compute | False | 0.00 | 0.00 |
| Fermyon Spin | light-compute | True | 0.00 | 0.00 |
| Fermyon Spin | query-external | False | 4.87 | 2.48 |
| Fermyon Spin | query-external | True | 5.15 | 2.55 |

## Key Observations

- Overhead related observations:
- AWS Lambda consistently shows significantly higher overhead during cold starts compared to warm starts, with cold start penalties ranging from 60% to over 170%.
- Fermyon Spin demonstrates minimal or even negative cold start penalties in some functions, indicating extremely low provisioning cost and potential performance stability during initial execution.
- Lighter functions like `basic-http`, `key-value`, and `light-compute` suffer disproportionately from AWS cold starts, leading to poor performance predictability in low-traffic or bursty scenarios.
- Fermyon’s overhead measurements are more stable across cold and warm runs, reflected in both lower cold start penalties and smaller standard deviation values — a strong indicator of consistent platform behavior.
- Even for compute-heavy tasks like `heavy-compute`, AWS cold start overhead remains high, suggesting that cold starts are a persistent bottleneck regardless of workload complexity.
- Fermyon appears particularly well-suited for latency-sensitive or on-demand workloads such as microservices, due to its near-zero cold start impact and predictable overhead latency.
- These trends strongly support Hypothesis 2 — that WebAssembly-based platforms like Fermyon Spin significantly reduce overhead latency and cold start delays compared to traditional serverless platforms.
- Total performance related observations:
- Fermyon consistently shows lower total time for lightweight functions like `basic-http`, due to much lower overhead.
- AWS sometimes outperforms Fermyon in compute-heavy tasks like `heavy-compute`, suggesting better execution scaling.
- The execution/overhead ratio is significantly higher on AWS for compute-heavy functions, but lower for I/O-bound ones.
- Fermyon maintains a more balanced ratio across functions, reflecting consistent low overhead and good runtime utilization.
- Cold starts impact AWS's total time more dramatically than Fermyon's, confirming prior observations from the overhead analysis.
- These results support the idea that Fermyon is optimal for latency-sensitive, short-lived workloads, while AWS may be better suited for heavier compute-bound workloads.
