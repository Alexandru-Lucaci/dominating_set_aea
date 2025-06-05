import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# First, let's load all the data
all_data = []
files_info = [
    ("resultsRenamed/bremen_subgraph_20.gr__ourSolution__data.csv", "Branch & Bound", 20),
    ("resultsRenamed/bremen_subgraph_20.gr__fast_tabu_search__data.csv", "Fast Tabu Search", 20),
    ("resultsRenamed/bremen_subgraph_20.gr__orTools__data.csv", "OR-Tools", 20),
    ("resultsRenamed/bremen_subgraph_20.gr__tabu_search__data.csv", "Tabu Search", 20),
    ("resultsRenamed/bremen_subgraph_50.gr__tabu_search__data.csv", "Tabu Search", 50),
    ("resultsRenamed/bremen_subgraph_50.gr__fast_tabu_search__data.csv", "Fast Tabu Search", 50),
    ("resultsRenamed/bremen_subgraph_50.gr__orTools__data.csv", "OR-Tools", 50),
    ("resultsRenamed/bremen_subgraph_50.gr__ourSolution__data.csv", "Branch & Bound", 50),
    ("resultsRenamed/bremen_subgraph_100.gr__ourSolution__data.csv", "Branch & Bound", 100),
    ("resultsRenamed/bremen_subgraph_100.gr__orTools__data.csv", "OR-Tools", 100),
    ("resultsRenamed/bremen_subgraph_100.gr__fast_tabu_search__data.csv", "Fast Tabu Search", 100),
    ("resultsRenamed/bremen_subgraph_100.gr__tabu_search__data.csv", "Tabu Search", 100),
    ("resultsRenamed/bremen_subgraph_150.gr__fast_tabu_search__data.csv", "Fast Tabu Search", 150),
    ("resultsRenamed/bremen_subgraph_150.gr__ourSolution__data.csv", "Branch & Bound", 150),
    ("resultsRenamed/bremen_subgraph_150.gr__tabu_search__data.csv", "Tabu Search", 150),
    ("resultsRenamed/bremen_subgraph_150.gr__orTools__data.csv", "OR-Tools", 150),
    ("resultsRenamed/bremen_subgraph_200.gr__fast_tabu_search__data.csv", "Fast Tabu Search", 200),
    ("resultsRenamed/bremen_subgraph_200.gr__ourSolution__data.csv", "Branch & Bound", 200),
    ("resultsRenamed/bremen_subgraph_200.gr__tabu_search__data.csv", "Tabu Search", 200),
    ("resultsRenamed/bremen_subgraph_200.gr__orTools__data.csv", "OR-Tools", 200),
    ("resultsRenamed/bremen_subgraph_250.gr__tabu_search__data.csv", "Tabu Search", 250),
    ("resultsRenamed/bremen_subgraph_250.gr__fast_tabu_search__data.csv", "Fast Tabu Search", 250),
    ("resultsRenamed/bremen_subgraph_250.gr__ourSolution__data.csv", "Branch & Bound", 250),
    ("resultsRenamed/bremen_subgraph_250.gr__orTools__data.csv", "OR-Tools", 250),
    ("resultsRenamed/bremen_subgraph_300.gr__fast_tabu_search__data.csv", "Fast Tabu Search", 300),
    ("resultsRenamed/bremen_subgraph_300.gr__orTools__data.csv", "OR-Tools", 300),
    ("resultsRenamed/bremen_subgraph_300.gr__tabu_search__data.csv", "Tabu Search", 300),
    ("resultsRenamed/bremen_subgraph_300.gr__ourSolution__data.csv", "Branch & Bound", 300),
]

for filename, solver, graph_size in files_info:
    try:
        df = pd.read_csv(filename)
        df['Solver'] = solver
        df['Graph_Size'] = graph_size
        df['Quality_Deviation'] = ((df['Number of Vertices'] - df['Number of vertices expected']) /
                                   df['Number of vertices expected'] * 100)
        all_data.append(df)
    except Exception as e:
        print(f"Error reading {filename}: {e}")

combined_df = pd.concat(all_data, ignore_index=True)

# Create a comprehensive hypothesis testing report
hypothesis_tests = []

print("="*80)
print("HYPOTHESIS TESTING FOR ALGORITHM PERFORMANCE")
print("="*80)

# 1. PARAMETRIC TESTS

print("\n1. PARAMETRIC TESTS")
print("-"*40)

# 1.1 Two-sample t-test: Compare Tabu Search vs Fast Tabu Search
print("\n1.1 Two-sample t-test: Tabu Search vs Fast Tabu Search")
print("H0: Mean quality deviation is equal for both algorithms")
print("H1: Mean quality deviation is different")

tabu_quality = combined_df[combined_df['Solver'] == 'Tabu Search']['Quality_Deviation']
fast_tabu_quality = combined_df[combined_df['Solver'] == 'Fast Tabu Search']['Quality_Deviation']

t_stat, p_value = stats.ttest_ind(tabu_quality, fast_tabu_quality)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} at α=0.05")

hypothesis_tests.append({
    'Test': 'Two-sample t-test',
    'Comparison': 'Tabu vs Fast Tabu (Quality)',
    'H0': 'Equal means',
    't-statistic': t_stat,
    'p-value': p_value,
    'Significant (α=0.05)': p_value < 0.05
})

# 1.2 Paired t-test: Compare execution times for same graph sizes
print("\n1.2 Paired t-test: Tabu Search vs Fast Tabu Search (same graphs)")
print("H0: Mean time difference is zero")
print("H1: Mean time difference is not zero")

# Get paired data for same graph sizes
paired_data = []
for size in [20, 50, 100, 150, 200, 250, 300]:
    tabu_times = combined_df[(combined_df['Solver'] == 'Tabu Search') &
                            (combined_df['Graph_Size'] == size)]['Time'].values
    fast_tabu_times = combined_df[(combined_df['Solver'] == 'Fast Tabu Search') &
                                  (combined_df['Graph_Size'] == size)]['Time'].values

    if len(tabu_times) > 0 and len(fast_tabu_times) > 0:
        # Take mean for each graph size
        paired_data.append((tabu_times.mean(), fast_tabu_times.mean()))

if len(paired_data) > 0:
    tabu_paired = [x[0] for x in paired_data]
    fast_tabu_paired = [x[1] for x in paired_data]

    t_stat_paired, p_value_paired = stats.ttest_rel(tabu_paired, fast_tabu_paired)
    print(f"t-statistic: {t_stat_paired:.4f}")
    print(f"p-value: {p_value_paired:.4f}")
    print(f"Result: {'Reject H0' if p_value_paired < 0.05 else 'Fail to reject H0'} at α=0.05")

    hypothesis_tests.append({
        'Test': 'Paired t-test',
        'Comparison': 'Tabu vs Fast Tabu (Time)',
        'H0': 'Mean difference = 0',
        't-statistic': t_stat_paired,
        'p-value': p_value_paired,
        'Significant (α=0.05)': p_value_paired < 0.05
    })

# 1.3 One-sample t-test: Test if Branch & Bound deviation is significantly different from 10%
print("\n1.3 One-sample t-test: Branch & Bound quality deviation vs 10%")
print("H0: Mean quality deviation = 10%")
print("H1: Mean quality deviation ≠ 10%")

bb_quality = combined_df[combined_df['Solver'] == 'Branch & Bound']['Quality_Deviation']
t_stat_one, p_value_one = stats.ttest_1samp(bb_quality, 10)
print(f"t-statistic: {t_stat_one:.4f}")
print(f"p-value: {p_value_one:.4f}")
print(f"Result: {'Reject H0' if p_value_one < 0.05 else 'Fail to reject H0'} at α=0.05")

hypothesis_tests.append({
    'Test': 'One-sample t-test',
    'Comparison': 'Branch & Bound vs 10%',
    'H0': 'Mean = 10%',
    't-statistic': t_stat_one,
    'p-value': p_value_one,
    'Significant (α=0.05)': p_value_one < 0.05
})

# 2. NON-PARAMETRIC TESTS

print("\n\n2. NON-PARAMETRIC TESTS")
print("-"*40)

# 2.1 Mann-Whitney U test: Compare OR-Tools vs all others
print("\n2.1 Mann-Whitney U test: OR-Tools vs Other Algorithms (Quality)")
print("H0: Distributions are equal")
print("H1: Distributions are different")

ortools_quality = combined_df[combined_df['Solver'] == 'OR-Tools']['Quality_Deviation']
others_quality = combined_df[combined_df['Solver'] != 'OR-Tools']['Quality_Deviation']

u_stat, p_value_mw = stats.mannwhitneyu(ortools_quality, others_quality, alternative='two-sided')
print(f"U-statistic: {u_stat:.4f}")
print(f"p-value: {p_value_mw:.4f}")
print(f"Result: {'Reject H0' if p_value_mw < 0.05 else 'Fail to reject H0'} at α=0.05")

hypothesis_tests.append({
    'Test': 'Mann-Whitney U',
    'Comparison': 'OR-Tools vs Others (Quality)',
    'H0': 'Equal distributions',
    'U-statistic': u_stat,
    'p-value': p_value_mw,
    'Significant (α=0.05)': p_value_mw < 0.05
})

# 2.2 Wilcoxon signed-rank test: Compare Tabu vs Fast Tabu on paired data
print("\n2.2 Wilcoxon signed-rank test: Tabu vs Fast Tabu (Quality, paired)")
print("H0: Median difference = 0")
print("H1: Median difference ≠ 0")

# Get quality differences for same instances
quality_diffs = []
for size in [20, 50, 100, 150, 200, 250, 300]:
    tabu_q = combined_df[(combined_df['Solver'] == 'Tabu Search') &
                         (combined_df['Graph_Size'] == size)]['Quality_Deviation'].values
    fast_tabu_q = combined_df[(combined_df['Solver'] == 'Fast Tabu Search') &
                              (combined_df['Graph_Size'] == size)]['Quality_Deviation'].values

    if len(tabu_q) > 0 and len(fast_tabu_q) > 0:
        quality_diffs.append(tabu_q.mean() - fast_tabu_q.mean())

if len(quality_diffs) > 1:
    w_stat, p_value_w = stats.wilcoxon(quality_diffs)
    print(f"W-statistic: {w_stat:.4f}")
    print(f"p-value: {p_value_w:.4f}")
    print(f"Result: {'Reject H0' if p_value_w < 0.05 else 'Fail to reject H0'} at α=0.05")

    hypothesis_tests.append({
        'Test': 'Wilcoxon signed-rank',
        'Comparison': 'Tabu vs Fast Tabu (Quality diff)',
        'H0': 'Median difference = 0',
        'W-statistic': w_stat,
        'p-value': p_value_w,
        'Significant (α=0.05)': p_value_w < 0.05
    })

# 2.3 Kruskal-Wallis test: Compare all algorithms
print("\n2.3 Kruskal-Wallis test: All algorithms (Quality)")
print("H0: All algorithms have the same distribution")
print("H1: At least one algorithm differs")

groups = []
for solver in ['Branch & Bound', 'OR-Tools', 'Tabu Search', 'Fast Tabu Search']:
    groups.append(combined_df[combined_df['Solver'] == solver]['Quality_Deviation'].values)

h_stat, p_value_kw = stats.kruskal(*groups)
print(f"H-statistic: {h_stat:.4f}")
print(f"p-value: {p_value_kw:.4f}")
print(f"Result: {'Reject H0' if p_value_kw < 0.05 else 'Fail to reject H0'} at α=0.05")

hypothesis_tests.append({
    'Test': 'Kruskal-Wallis',
    'Comparison': 'All algorithms (Quality)',
    'H0': 'Equal distributions',
    'H-statistic': h_stat,
    'p-value': p_value_kw,
    'Significant (α=0.05)': p_value_kw < 0.05
})

# Save hypothesis test results
hypothesis_df = pd.DataFrame(hypothesis_tests)
hypothesis_df.to_csv('hypothesis_test_results.csv', index=False)

# Create confidence intervals
print("\n\n3. CONFIDENCE INTERVALS (95%)")
print("-"*40)

ci_results = []
for solver in ['Branch & Bound', 'OR-Tools', 'Tabu Search', 'Fast Tabu Search']:
    solver_data = combined_df[combined_df['Solver'] == solver]

    # Quality deviation CI
    quality = solver_data['Quality_Deviation']
    n = len(quality)
    mean_q = quality.mean()
    sem_q = stats.sem(quality)
    ci_q = stats.t.interval(0.95, n-1, loc=mean_q, scale=sem_q)

    # Time CI
    time = solver_data['Time']
    mean_t = time.mean()
    sem_t = stats.sem(time)
    ci_t = stats.t.interval(0.95, n-1, loc=mean_t, scale=sem_t)

    print(f"\n{solver}:")
    print(f"  Quality Deviation: {mean_q:.2f}% [{ci_q[0]:.2f}%, {ci_q[1]:.2f}%]")
    print(f"  Execution Time: {mean_t:.2f}s [{ci_t[0]:.2f}s, {ci_t[1]:.2f}s]")

    ci_results.append({
        'Solver': solver,
        'Quality_Mean': mean_q,
        'Quality_CI_Lower': ci_q[0],
        'Quality_CI_Upper': ci_q[1],
        'Time_Mean': mean_t,
        'Time_CI_Lower': ci_t[0],
        'Time_CI_Upper': ci_t[1]
    })

ci_df = pd.DataFrame(ci_results)
ci_df.to_csv('confidence_intervals.csv', index=False)

print("\n\nFiles generated:")
print("1. hypothesis_test_results.csv")
print("2. confidence_intervals.csv")