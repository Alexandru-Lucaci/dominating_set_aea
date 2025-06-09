# Dominating Set Solver

A comprehensive implementation of various algorithms for solving the **Minimum Dominating Set** problem, featuring exact and heuristic approaches with detailed performance analysis.

**First versions of the scripts can be found in the oldScripts directory and the newest scripts are in the home directory 

Also the documentation and what we presented in the last lab can be found in the Documentation.docx

The results of our lattest modifications can be found in the resultsRenamed folder **


## Problem Overview

The **Minimum Dominating Set (MDS)** problem is a fundamental NP-hard problem in graph theory. Given an undirected graph G = (V, E), a dominating set D ⊆ V is a subset of vertices such that every vertex in V is either in D or adjacent to at least one vertex in D. The goal is to find the smallest possible dominating set.

### Applications
- **Network Surveillance**: Monitoring network infrastructure
- **Resource Allocation**: Optimal placement of facilities or sensors
- **Social Network Analysis**: Identifying influential users
- **Wireless Sensor Networks**: Minimizing energy consumption

## Features

- **4 Different Algorithms**: Exact and heuristic approaches
- **Advanced Optimization**: Multiple bounding strategies and metaheuristics
- **Comprehensive Testing**: Benchmarked on graphs from 20 to 300+ vertices
- **Detailed Analytics**: Performance comparison and statistical analysis
- **Visualization Support**: Graph plotting with dominating set highlighting
- **PACE Format Support**: Compatible with standard competition formats

## Algorithm Performance Summary

| Algorithm | Accuracy | Speed | Best Use Case |
|-----------|----------|-------|---------------|
| **OR-Tools** | 🟢 Perfect (0% deviation) | 🟢 Fastest (0.49s avg) | **Recommended for all cases** |
| **Fast Tabu Search** | 🟡 Good (6.38% deviation) | 🟢 Fast (32.25s avg) | Large graphs, time-critical |
| **Tabu Search** | 🟢 Excellent (2.85% deviation) | 🟡 Moderate (255.94s avg) | Quality over speed |
| **Branch & Bound** | 🔴 Poor (11.80% deviation) | 🔴 Slow (307.71s avg) | Small graphs only |

## Installation

### Prerequisites
```bash
pip install ortools matplotlib networkx pandas numpy seaborn
```

### Clone Repository
```bash
git clone https://github.com/yourusername/dominating-set-solver.git
cd dominating-set-solver
```

## Quick Start

### Basic Usage

```python
from graph import Graph
from utils.parser import parse_pace_input
from solvers.ortools_solver import ORToolsDominatingSetSolver

# Load graph from PACE format
n, edges = parse_pace_input("path/to/graph.gr")
graph = Graph(n)
for u, v in edges:
    graph.add_edge(u-1, v-1)  # Convert to 0-based indexing

# Solve using OR-Tools (recommended)
solver = ORToolsDominatingSetSolver(graph)
solver.build_model()
solution = solver.solve(time_limit=300)

print(f"Found dominating set of size {len(solution)}: {solution}")
```

### Algorithm Comparison

```python
from solvers.bnb_solver import BranchAndBoundDominatingSetSolver
from solvers.tabu_solver import tabu_search_dominating_set
from strategies.bounding import SimpleBound
import time

# Test all algorithms
algorithms = {}

# OR-Tools
start = time.time()
ortools_solver = ORToolsDominatingSetSolver(graph)
ortools_solver.build_model()
ortools_solution = ortools_solver.solve(time_limit=300)
algorithms['OR-Tools'] = {
    'solution': ortools_solution,
    'time': time.time() - start,
    'size': len(ortools_solution)
}

# Fast Tabu Search
start = time.time()
tabu_solution = tabu_search_dominating_set(
    graph.adjacency_list,
    time_limit=300
)
algorithms['Tabu Search'] = {
    'solution': tabu_solution,
    'time': time.time() - start,
    'size': len(tabu_solution)
}

# Compare results
for name, result in algorithms.items():
    print(f"{name}: Size {result['size']}, Time {result['time']:.2f}s")
```

## Available Algorithms

### 1. OR-Tools CP-SAT Solver - **RECOMMENDED**
```python
from solvers.ortools_solver import ORToolsDominatingSetSolver

solver = ORToolsDominatingSetSolver(graph)
solver.build_model()
solution = solver.solve(time_limit=300)
```
- **Always finds optimal solutions**
- **Fastest execution time** (0.49s average)
- **Scales well** to medium-large graphs
- **Production ready**

### 2. Fast Tabu Search
```python
from solvers.tabu_solver import tabu_search_dominating_set_optimized

solution = tabu_search_dominating_set_optimized(
    graph.adjacency_list,
    max_iterations=5000,
    time_limit=300,
    adaptive_tenure=True
)
```
- **Good solution quality** (6.38% deviation average)
- **Fast execution** (32.25s average)
- **Excellent for large graphs**
- Near-optimal, not guaranteed optimal

### 3. Standard Tabu Search
```python
solution = tabu_search_dominating_set(
    graph.adjacency_list,
    max_iterations=1000,
    tabu_tenure=25,
    time_limit=300
)
```
- **High solution quality** (2.85% deviation average)
- Slower execution time
- Diminishing returns vs Fast Tabu

### 4. Branch and Bound
```python
from solvers.bnb_solver import BranchAndBoundDominatingSetSolver
from strategies.bounding import StrongBound

solver = BranchAndBoundDominatingSetSolver(
    graph,
    StrongBound(),
    time_limit=300
)
solution = solver.solve()
```
- **Theoretically exact** (if completes)
- Poor performance in practice
- Often times out on larger graphs
- **For research/educational purposes only**

## Performance Analysis

### Scalability by Graph Size

| Graph Size | OR-Tools | Fast Tabu | Tabu Search | Branch & Bound |
|------------|----------|-----------|-------------|----------------|
| 20 vertices | 0.22s | 5.60s | 0.00s | 80.84s |
| 50 vertices | 0.32s | 18.73s | 298.33s | 99.25s |
| 100 vertices | 0.40s | 34.72s | 298.38s | 192.82s |
| 150 vertices | 0.43s | 41.36s | 298.38s | 229.06s |
| 200 vertices | 0.47s | 39.45s | 298.58s | 502.63s |
| 300 vertices | 1.01s | 43.03s | 299.04s | 529.97s |

### Algorithm Selection Guide

```python
def recommend_algorithm(graph_size, time_limit, quality_requirement):
    """Smart algorithm selection based on constraints"""

    if quality_requirement == "optimal":
        return "OR-Tools"

    if time_limit < 60:  # Less than 1 minute
        if graph_size < 100:
            return "OR-Tools"
        else:
            return "Fast Tabu Search"

    if graph_size > 200:
        return "Fast Tabu Search"

    if quality_requirement == "high":
        return "Tabu Search"

    return "OR-Tools"  # Default recommendation
```

## Project Structure

```
dominating-set-solver/
├── solvers/
│   ├── bnb_solver.py          # Branch and Bound implementation
│   ├── ortools_solver.py      # OR-Tools CP-SAT solver
│   └── tabu_solver.py         # Tabu Search variants
├── strategies/
│   └── bounding.py            # Bounding strategies for B&B
├── utils/
│   ├── parser.py              # PACE format parser
│   ├── validator.py           # Solution validation
│   └── visualization.py       # Graph plotting
├── graph.py                   # Graph data structure
├── logger.py                  # Logging utilities
├── main.py                    # Batch testing framework
└── visualization-script.py    # Performance analysis
```

## Running Experiments

### Single Test Case
```bash
python main.py --orTools --numberOfRuns 1 --timeLimit 300
```

### Full Benchmark Suite
```bash
python main.py --orTools --tabu_search --ourSolution --numberOfRuns 5 --timeLimit 1800
```

### Custom Testing
```python
from main import run_single_case

# Test specific graph with multiple solvers
run_single_case(
    run_index=1,
    testFile="bremen_subgraph_100.gr",
    solFile="bremen_subgraph_100.sol",
    testFileDir="path/to/test/files",
    timeLimit=300,
    orTools=True,
    use_tabu=True,
    ourSolution=False
)
```

## Visualization

### Generate Performance Plots
```bash
python visualization-script.py results/ output_visualizations/
```

### Create Solution Visualization
```python
from utils.visualization import draw_graph

draw_graph(
    testFilePath="bremen_subgraph_100.gr",
    dominating_set=[0, 5, 12, 18, 25],  # 0-based vertex indices
    title="Bremen Subgraph 100",
    filePath="solution_visualization.png"
)
```

## Input Format

The solver accepts graphs in **PACE format**:

```
c This is a comment line
c Graph with 5 vertices and 6 edges
p ds 5 6
1 2
1 3
2 4
3 4
4 5
2 5
```

Where:
- Lines starting with `c` are comments
- `p ds n m` indicates n vertices and m edges
- Each subsequent line represents an edge (1-indexed)

## Recommendations

### For Production Use
```python
# Always use OR-Tools first
solver = ORToolsDominatingSetSolver(graph)
solver.build_model()
solution = solver.solve(time_limit=300)

if solution:  # OR-Tools found solution
    return solution
else:  # Fallback to Fast Tabu Search
    return tabu_search_dominating_set_optimized(
        graph.adjacency_list,
        time_limit=300
    )
```

### For Research
- **Small graphs (≤100 vertices)**: Branch and Bound for educational purposes
- **Algorithm comparison**: Test multiple approaches with statistical analysis
- **New heuristics**: Extend Tabu Search or implement genetic algorithms

### For Large-Scale Applications
- **Always start with OR-Tools**
- **Use Fast Tabu Search for graphs >300 vertices**
- **Consider graph decomposition for massive instances**

## Statistical Analysis

The project includes comprehensive statistical testing:

- **Parametric tests**: t-tests for mean comparisons
- **Non-parametric tests**: Mann-Whitney U, Kruskal-Wallis
- **Confidence intervals**: 95% CI for all performance metrics
- **Effect size analysis**: Cohen's d for practical significance
