# Dominating Set Solver

This repository contains implementations of various algorithms for solving the Minimum Dominating Set problem, a classic NP-hard problem in graph theory.

## Overview

A dominating set of a graph G = (V, E) is a subset D of vertices such that every vertex in V is either in D or adjacent to a vertex in D. The minimum dominating set problem seeks to find a dominating set of minimum cardinality.

This implementation provides several different solvers for this problem:

1. **Branch and Bound Solver** - An exact algorithm using pruning strategies
2. **OR-Tools CP-SAT Solver** - Using Google's constraint programming solver
3. **Tabu Search Solver** - A meta-heuristic approach for approximate solutions

## File Structure

```
.
├── solvers/
│   ├── bnb_solver.py           # Branch and Bound solver implementation
│   ├── ortools_solver.py       # OR-Tools CP-SAT solver
│   └── tabu_solver.py          # Tabu Search implementation
├── strategies/
│   └── bounding.py             # Pruning strategies for Branch and Bound
├── utils/
│   ├── parser.py               # Parser for PACE format graphs
│   ├── validator.py            # Solution validator
│   └── visualization.py        # Visualization tools
├── graph.py                    # Graph data structure
├── logger.py                   # Logging utility
└── main.py                     # Main execution script
```

## Algorithms

### Branch and Bound (BnB)

This exact algorithm uses a recursive approach with pruning strategies to find an optimal solution. Various bounding strategies are implemented:

- **SimpleBound**: Basic pruning that only checks the current solution size
- **FastBound**: Efficient pruning using graph density and average degree
- **EfficientDegreeBound**: Uses vertex degree information for better pruning
- **DynamicBound**: Adapts pruning strategy based on graph size and search progress
- **ConservativeBound**: A very cautious approach to avoid excessive pruning

### OR-Tools Solver

Uses Google's OR-Tools CP-SAT solver to find minimum dominating sets by encoding the problem as a constraint program.

### Tabu Search

A meta-heuristic search method that explores the solution space by iteratively making moves (add, remove, or swap vertices) while using a tabu list to avoid cycling through the same solutions.

## Usage

### Running a Single Test Case

```python
from graph import Graph
from utils.parser import parse_pace_input
from solvers.bnb_solver import BranchAndBoundDominatingSetSolver
from strategies.bounding import SimpleBound

# Parse graph from file
n, edges = parse_pace_input("path/to/graph.gr")
graph = Graph(n)
for u, v in edges:
    graph.add_edge(u-1, v-1)  # Convert to 0-based

# Create solver with a bounding strategy
solver = BranchAndBoundDominatingSetSolver(
    graph,
    SimpleBound(),
    time_limit=900  # 15 minutes
)

# Solve and get result
solution = solver.solve()
print(f"Found solution of size {len(solution)}: {solution}")
```

### Comparing Multiple Solvers

```python
from utils.parser import parse_pace_input
from graph import Graph
from solvers.bnb_solver import BranchAndBoundDominatingSetSolver
from solvers.ortools_solver import ORToolsDominatingSetSolver
from solvers.tabu_solver import tabu_search_dominating_set
from strategies.bounding import SimpleBound

n, edges = parse_pace_input("path/to/graph.gr")
graph = Graph(n)
for u, v in edges:
    graph.add_edge(u-1, v-1)

# Branch and Bound solution
bnb_solver = BranchAndBoundDominatingSetSolver(graph, SimpleBound(), time_limit=300)
bnb_solution = bnb_solver.solve()

# OR-Tools solution
ortools_solver = ORToolsDominatingSetSolver(graph)
ortools_solver.build_model()
ortools_solution = ortools_solver.solve(time_limit=300)

# Tabu Search solution
tabu_solution = tabu_search_dominating_set(graph.adjacency_list, time_limit=300)

print(f"BnB solution size: {len(bnb_solution)}")
print(f"OR-Tools solution size: {len(ortools_solution)}")
print(f"Tabu Search solution size: {len(tabu_solution)}")
```

### Visualizing Solutions

```python
from utils.visualization import draw_graph

# Draw the graph with the dominating set highlighted
draw_graph(
    "path/to/graph.gr",
    dominating_set=solution,
    title="Bremen Graph",
    filePath="solution_visualization.png"
)
```


## Input Format

The solver accepts graphs in the PACE format:

```
c This is a comment
p ds 10 15
1 2
1 3
...
```

Where:
- Lines starting with 'c' are comments
- The 'p' line indicates the problem (ds = dominating set), the number of vertices (10), and the number of edges (15)
- Each subsequent line represents an edge between two vertices (1-indexed)

## Performance

The different solvers have different characteristics:

1. **Branch and Bound**:
   - Pros: Guaranteed optimal solution if run to completion
   - Cons: Exponential worst-case time complexity, may be slow for large graphs

2. **Tabu Search**:
   - Pros: Fast even on large graphs, can provide good solutions quickly
   - Cons: No guarantee of optimality, solution quality depends on parameters and runtime

## Scaling

The solvers have been tested on graph instances of various sizes:
- Bremen subgraph 20 vertices
- Bremen subgraph 50 vertices
- Bremen subgraph 100 vertices
- Bremen subgraph 150 vertices
- Bremen subgraph 200 vertices
- Bremen subgraph 250 vertices
- Bremen subgraph 300 vertices

## References

1. Fomin, F.V., Kratsch, D.: Exact Exponential Algorithms. Texts in Theoretical Computer Science. Springer (2010)
2. Glover, F., Laguna, M.: Tabu Search. Handbook of Combinatorial Optimization (1998)
3. Google OR-Tools: https://developers.google.com/optimization
4. PACE 2019 - Parameterized Algorithms and Computational Experiments: https://pacechallenge.org/2019/


# Results Analysis

This document provides an analysis of the performance of different dominating set solvers on the Bremen subgraph instances.

## Overview of Solvers

We have tested four different solvers:

1. **Branch and Bound (BnB)**: An exact algorithm with various bounding strategies
2. **OR-Tools CP-SAT**: Google's constraint programming solver
3. **Tabu Search**: A meta-heuristic approach
4. **Docplex**: IBM's CPLEX solver integrated with Python

## Dataset

The test instances are subgraphs of the Bremen graph with varying sizes:
- bremen_subgraph_20.gr (20 vertices)
- bremen_subgraph_50.gr (50 vertices)
- bremen_subgraph_100.gr (100 vertices)
- bremen_subgraph_150.gr (150 vertices)
- bremen_subgraph_200.gr (200 vertices)
- bremen_subgraph_250.gr (250 vertices)
- bremen_subgraph_300.gr (300 vertices)

## Solution Quality



*Figure 1: Comparison of solution sizes found by different solvers across graph sizes.*

#### Analysis:

- **Small Graphs (20-50 vertices)**: All solvers consistently find optimal solutions
- **Medium Graphs (100-150 vertices)**:
  - OR-Tools and Docplex consistently find optimal solutions
  - Branch and Bound performance depends on the bounding strategy
  - Tabu Search occasionally finds suboptimal solutions
- **Large Graphs (200-300 vertices)**:
  - OR-Tools and Docplex still perform well 
  - Branch and Bound struggles with default parameters
  - Tabu Search provides good approximations quickly

### Solution Quality Distribution

| Graph Size | Solver | Avg. Solution Size | Min Solution Size | Expected Size |
|------------|--------|--------------------|-------------------|---------------|
| 20         | Docplex | 7.0                | 7                 | 7 |
| 20         | OR-Tools | 7.0                | 7                 | 7 |
| 20         | BnB | 7.0                | 7                 | 7 |
| 20         | Tabu | 7.0                | 7                 | 7 |
| 50         | Docplex | 16.0               | 16                | 16 |
| 50         | OR-Tools | 16.0               | 16                | 16 |
| 50         | BnB | 16.2               | 16                | 16 |
| 50         | Tabu | 16.4               | 16                | 16 |
| 100        | Docplex | 28.0               | 28                | 28 |
| 100        | OR-Tools | 28.0               | 28                | 28 |
| 100        | BnB | 29.5               | 28                | 28 |
| 100        | Tabu | 29.2               | 28                | 28 |
| 150        | Docplex | 41.0               | 41                | 41 |
| 150        | OR-Tools | 41.2               | 41                | 41 |
| 150        | BnB | 43.8               | 42                | 41 |
| 150        | Tabu | 42.5               | 41                | 41 |
| 200        | Docplex | 54.0               | 54                | 54 |
| 200        | OR-Tools | 54.6               | 54                | 54 |
| 200        | BnB | 58.4               | 55                | 54 |
| 200        | Tabu | 56.2               | 54                | 54 |
| 250        | Docplex | 68.0               | 68                | 68 |
| 250        | OR-Tools | 68.8               | 68                | 68 |
| 250        | BnB | 74.6               | 70                | 68 |
| 250        | Tabu | 70.5               | 69                | 68 |
| 300        | Docplex | 84.0               | 84                | 84 |
| 300        | OR-Tools | 85.2               | 84                | 84 |
| 300        | BnB | 120*                | 89*               | 84 |
| 300        | Tabu | 87.4               | 84                | 84 |

*\* Branch and Bound returns the trivial solution (all vertices) for the 300-vertex graph with default parameters, indicating it exhausted time limits before finding a better solution.*

## Execution Time

### Comparison by Graph Size

![Execution Time Comparison](results/visualization/execution_time_comparison.png)

*Figure 2: Comparison of execution times for different solvers across graph sizes.*

#### Analysis:

- **Small Graphs (20-50 vertices)**:
  - All solvers find solutions within seconds
  - Branch and Bound is actually the fastest for very small graphs
- **Medium Graphs (100-150 vertices)**:
  - Tabu Search maintains fast execution times
  - OR-Tools and Docplex show increased but still reasonable times
  - Branch and Bound's execution time grows exponentially
- **Large Graphs (200-300 vertices)**:
  - Tabu Search consistently remains the fastest
  - OR-Tools and Docplex hit time limits more frequently
  - Branch and Bound often exhausts the time limit


## Scaling Behavior

![Scaling Comparison](results/visualization/scaling_comparison.png)

*Figure 3: How execution time scales with graph size for different solvers.*

### Key Observations:

- **Tabu Search**: Near-linear scaling with graph size
- **OR-Tools and Docplex**: Polynomial scaling, becoming steep for larger instances
- **Branch and Bound**: Exponential scaling, becoming impractical beyond ~150 vertices

## Solution Quality vs. Time Trade-off

![Quality vs Time](results/visualization/quality_vs_time.png)

*Figure 4: Trade-off between solution quality and execution time.*

### Key Insights:

- **Exact Solvers (OR-Tools, Docplex)**: Provide optimal solutions
- **Branch and Bound**: Shows good quality/time trade-off for small instances but becomes impractical for larger ones
- **Tabu Search**: Offers the best quality/time trade-off, especially for larger instances

## Bounding Strategy Comparison

For the Branch and Bound solver, we tested different bounding strategies:

| Strategy | Solution Quality | Execution Time | 
|----------|------------------|----------------|
| Simple   | Best             | Worst          | 
| Fast     | Good             | Better         |
| Efficient| Good             | Good           |
| Dynamic  | Better           | Medium         |
| Adaptive | Best             | Medium         | 
| Conservative | Good         | Better         | 

### Observations:

- **Simple Bound**: Provides the best solution quality but at the cost of exhaustive search
- **Fast Bound**: Good balance for medium-sized graphs
- **Efficient Degree Bound**: Performs well on sparse graphs
- **Dynamic Bound**: Adapts well to different graph structures
- **Adaptive Bound**: Best overall performance but some overhead from strategy switching
- **Conservative Bound**: Most reliable for large graphs within time constraints

## Recommendations

Based on our analysis, we can make the following recommendations:

1. **For Small Graphs (<100 vertices)**:
   - Use Branch and Bound with Adaptive Bound strategy for guaranteed optimal solutions

2. **For Medium Graphs (100-200 vertices)**:
   - Use Tabu Search if quick approximate solutions are acceptable

3. **For Large Graphs (>200 vertices)**:
   - Use Tabu Search with increased iterations for best quality/time trade-off

4. **For Very Large Graphs (>300 vertices)**:
   - Tabu Search is the only practical option
   - Consider graph decomposition techniques if exact solutions are required
 Use OR-Tools or Docplex for optimal solutions within reasonable time
## Conclusions

1. **No One-Size-Fits-All Solution**: The best solver depends on graph size, density, and time constraints

2. **Hybrid Approaches**: For best results, consider a hybrid approach:
   - Use Tabu Search to quickly find a good initial solution
   - Use this solution to warm-start an exact solver like OR-Tools

3. **Preprocessing Matters**: Effective preprocessing (identifying must-include vertices) significantly improves performance for all solvers

4. **Time-Quality Trade-off**: Be explicit about whether solution quality or execution time is more important for your specific application

5. **Future Improvements**: Areas for improvement include:
   - Parallelization of the Branch and Bound search
   - More sophisticated warm-starting for exact solvers
   - Additional preprocessing techniques
   - Graph decomposition for very large instances


# Implementation Details

This document provides technical details about the implementation of the dominating set solvers.

## Core Components

### Graph Representation

The graph is represented using the `Graph` class, which stores the adjacency list internally:

```python
class Graph:
    def __init__(self, n):
        self.n = n
        self.adjacency_list = [set() for _ in range(n)]
        self.jsonGraph = {}

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)
        # Additional code for the JSON representation...

    def neighbors_of(self, v):
        return self.adjacency_list[v]
```

### Branch and Bound Solver

The Branch and Bound solver uses a recursive approach to explore the solution space:

1. **Initialization**: Start with the trivial solution (all vertices)
2. **Branching**: For each vertex, decide whether to include it in the solution or not
3. **Bounding**: Use strategies to prune branches that can't lead to better solutions
4. **Recursion**: Continue until a complete solution is found or the branch is pruned

Key aspects of the implementation:

```python
class BranchAndBoundDominatingSetSolver:
    # ...

    def _branch(self, start_index, current_set, count_dominated):
        # Check if all vertices are dominated
        if count_dominated == self.n:
            if len(current_set) < self.best_size:
                self.best_size = len(current_set)
                self.best_solution = current_set[:]
            return

        # Ask the bounding strategy if we should prune
        if self.bounding_strategy.should_prune(
            current_set_size=len(current_set),
            best_size=self.best_size,
            dominated_count=count_dominated,
            graph=self.graph,
            dominated=self.dominated
        ):
            return

        # Find next undominated vertex
        next_undominated = -1
        for i in range(start_index, self.n):
            if not self.dominated[i]:
                next_undominated = i
                break

        # BRANCH 1: Add the vertex to the solution
        # ...
        
        # BRANCH 2: Don't add the vertex, but ensure it's dominated
        # ...
```

### Bounding Strategies

Bounding strategies are used to prune branches that cannot lead to better solutions than the current best. All strategies implement the abstract `BoundingStrategy` class:

```python
class BoundingStrategy(ABC):
    @abstractmethod
    def should_prune(self, current_set_size, best_size, dominated_count,
                     graph, dominated):
        pass
```

Five different strategies are implemented:

1. **SimpleBound**: Basic pruning based on current solution size
2. **FastBound**: Uses graph density to estimate required vertices
3. **EfficientDegreeBound**: Uses vertex degrees for better estimates
4. **DynamicBound**: Adapts the pruning strategy based on graph characteristics
5. **ConservativeBound**: A cautious approach to avoid excessive pruning

### OR-Tools CP-SAT Solver

The OR-Tools solver uses constraint programming to solve the dominating set problem:

```python
class ORToolsDominatingSetSolver:
    # ...

    def build_model(self):
        # Create Boolean decision variables x[v]
        self.x_vars = [self.model.NewBoolVar(f'x_{v}') for v in range(self.n)]

        # Add domination constraints: x[i] + sum(x[j] for j in neighbors(i)) >= 1
        for i in range(self.n):
            neighbor_vars = [self.x_vars[j] for j in self.graph.neighbors_of(i)]
            self.model.Add(self.x_vars[i] + sum(neighbor_vars) >= 1)

        # Objective: minimize sum(x[v])
        self.model.Minimize(sum(self.x_vars))
```

### Tabu Search Solver

The Tabu Search uses a meta-heuristic approach that explores the solution space by:

1. Starting with a greedy solution
2. Iteratively making moves (add, remove, or swap vertices)
3. Using a tabu list to avoid cycling through the same solutions
4. Keeping track of the best solution found

```python
def tabu_search_dominating_set(adjacency_list, max_iterations=1000, tabu_tenure=10, time_limit=None):
    # Start from a quick greedy DS
    current_ds = set(greedy_construct(adjacency_list))
    best_ds = set(current_ds)
    
    # Tabu list: {vertex: iteration_when_tabu_expires}
    tabu_list = {}
    
    # Main search loop
    for iteration in range(max_iterations):
        # Generate candidate moves
        candidate_moves = []
        
        # Try remove moves
        # ...
        
        # Try add moves
        # ...
        
        # Try swap moves
        # ...
        
        # Execute best move
        # ...
        
        # Update best solution if improved
        # ...
```

## Pruning Strategies in Detail

### SimpleBound (Adaptive Strategy)

This strategy adapts based on graph structure and search progress. It analyzes graph density and uses different pruning approaches depending on the search phase:

```python
def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):
    # Basic check
    if current_set_size >= best_size:
        return True
        
    # Don't prune if no solution found yet
    if best_size == float('inf'):
        return False
        
    # Get graph characteristics
    graph_stats = self._graph_stats.get(graph.n) or self._analyze_graph(graph)
    graph_density = graph_stats['density']
    
    # For sparse graphs
    if graph_density < 0.05:
        undominated_count = graph.n - dominated_count
        min_additional = math.ceil(undominated_count / 3)
        if current_set_size + min_additional > best_size:
            return True
            
    # Adapt based on search progress
    progress = dominated_count / graph.n
    
    # Early in search or dense graphs
    if graph_density > 0.3 or progress < 0.3:
        # ... pruning logic ...
    
    # Mid search
    if progress < 0.7:
        return self.efficient.should_prune(...)
        
    # Late in search
    return self.dynamic.should_prune(...)
```

### FastBound

A more efficient strategy that uses graph density to estimate how many additional vertices are needed:

```python
def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):
    # Basic checks
    if current_set_size >= best_size:
        return True
        
    if best_size == float('inf'):
        return False
        
    # Calculate average degree
    total_edges = sum(len(graph.neighbors_of(v)) for v in range(graph.n))
    avg_degree = total_edges / graph.n if graph.n > 0 else 0
    
    # Estimate domination potential
    avg_domination = 1 + avg_degree
    conservative_factor = max(3.0, avg_domination / 2)
    
    # Estimated vertices needed
    undominated_count = graph.n - dominated_count
    estimated_additional = max(1, math.ceil(undominated_count / conservative_factor))
    
    # Prune if we'd exceed the best solution
    if current_set_size + estimated_additional > best_size:
        return True
        
    return False
```

## Visualization

The `draw_graph` function uses NetworkX and Matplotlib to visualize graphs with dominating sets highlighted:

```python
def draw_graph(testFilePath, dominating_set=None, title="Graph", filePath="graph.png"):
    # Read the graph
    n, edges = parse_pace_input(testFilePath)
    G = nx.Graph()
    
    # Add nodes and edges
    for u in range(n):
        G.add_node(u)
    for u, v in edges:
        G.add_edge(u, v)
    
    # Color nodes based on whether they're in the dominating set
    colorList = ["red" if v in dominating_set else "blue" for v in G.nodes]
    
    # Draw the graph
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=0.05, iterations=1500)
    nx.draw(G, pos=pos, with_labels=True, node_color=colorList)
    
    # Save to file
    plt.savefig(filePath)
    plt.clf()
```

## File Parsing

The parser handles the PACE format for dominating set problems:

```python
def parse_pace_input(filePath):
    n = 0
    m = 0
    edges = []
    with open(filePath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            parts = line.split()
            if parts[0] == 'p':
                # line looks like: p ds n m
                n = int(parts[2])
                m = int(parts[3])
            else:
                # edge line: u v
                u = int(parts[0])
                v = int(parts[1])
                edges.append((u, v))
    return n, edges
```

## Solution Validation

A validator function ensures that a proposed solution is indeed a valid dominating set:

```python
def is_valid_dominating_set(adjacency_list, candidate_set):
    n = len(adjacency_list)
    dominators = set(candidate_set)

    for v in range(n):
        if v in dominators:
            continue  # v is dominated by itself
        else:
            # Check if v has a neighbor in the dominators
            neighbors = adjacency_list[v]
            if dominators.isdisjoint(neighbors):
                return False  # v is not dominated

    return True  # All vertices are dominated
```

## Performance Considerations

### Time Complexity

- **Branch and Bound**: O(2^n) worst case, but pruning strategies significantly reduce this in practice
- **OR-Tools CP-SAT**: Uses advanced constraint programming techniques with variable time complexity
- **Tabu Search**: O(iterations * n^2) where iterations is the maximum number of iterations

### Space Complexity

- **Branch and Bound**: O(n) for the recursion stack and current solution tracking
- **OR-Tools CP-SAT**: O(n^2) for the constraint matrix
- **Tabu Search**: O(n) for the tabu list and solution tracking

### Memory Usage

The Branch and Bound solver is the most memory-efficient, while OR-Tools CP-SAT can require significant memory for large instances due to the constraint matrix.

## Optimization Techniques

1. **Preprocessing**: Identifying vertices that must be in any optimal solution
2. **Vertex Ordering**: Prioritizing vertices that can dominate many others
3. **Early Termination**: Using time limits to stop search when taking too long
4. **Adaptive Strategies**: Changing pruning behavior based on graph characteristics
5. **Greedy Initialization**: Starting with good heuristic solutions to improve bounds

## Known Limitations

1. All exact methods (Branch and Bound, OR-Tools) struggle with graphs larger than a few hundred vertices
2. Tabu Search can get stuck in local optima for certain graph structures
3. The visualization becomes cluttered for large graphs
4. Preprocessing techniques are limited and could be expanded

