import sys
import os
import subprocess
import time
import logging
from abc import ABC, abstractmethod
import csv
import argparse
import concurrent.futures
import multiprocessing
try:
    import resource
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "resource"])
    # import resource

import random
try:
    from ortools.sat.python import cp_model
except ImportError:
    subprocess.run(["pip", "install", "ortools"])
    from ortools.sat.python import cp_model
TEST_FILE_DIRECTORY = "ds_verifier/Dominating Set Verifier/src/test/resources/testset"
log_file_name = "log.txt"
loggingLevel = logging.INFO


class Logger:
    def __init__(self, log_file_name):
        self.log_file_name = log_file_name
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.handler = logging.FileHandler(log_file_name)
        self.handler.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

    def log(self, message:str, level=logging.INFO):
        if level == logging.INFO:
            self.logger.info(message)
            print(f"[INFO] [{time.strftime('%d-%m %H:%M:%S', time.localtime())}] : {message}")
        elif level == logging.ERROR :
            self.logger.error(message)
        elif level == logging.WARNING and loggingLevel == logging.WARNING:
            print(f"[WARNING] [{time.strftime('%d-%m %H:%M:%S', time.localtime())}] : {message}")
        # elif level == logging.WARNING:
        #     self.logger.warning(message)
    def close(self):
        self.handler.close()

class Graph:
    def __init__(self, n):
        """
        Create a graph with n vertices (0-based).
        Adjacency is stored in a list of sets for quick neighbor lookup.
        """
        self.n = n
        self.adjacency_list = [set() for _ in range(n)]
        self.jsonGraph = {}

    def add_edge(self, u, v):
        """
        Add undirected edge (u, v).
        u, v are assumed to be 0-based indices.
        """
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)
        u=u+1
        v=v+1
        if self.jsonGraph.get(u) is None:
            self.jsonGraph[u] = []
            self.jsonGraph[u].append(v)
        else:
            self.jsonGraph[u].append(v)
        if self.jsonGraph.get(v) is None:
            self.jsonGraph[v] = []
            self.jsonGraph[v].append(u)
        else:
            self.jsonGraph[v].append(u)

    def neighbors_of(self, v):
        """
        Return the set of neighbors of vertex v.
        """
        return self.adjacency_list[v]

class TestFile:
    def __init__(self, test_file:str, sol_file:str):
        self.test_file = test_file
        self.sol_file = sol_file

    def get_test_files(self):
        return self.test_files

    def get_sol_files(self):
        return self.sol_files

    def __repr__(self):
        return str(self.test_sol_map)


class BoundingStrategy(ABC):
    @abstractmethod
    def should_prune(self, current_set_size, best_size, dominated_count,
                     graph, dominated):
        """
        Decide whether to prune the current branch, given:
          - current_set_size: size of the partial dominating set
          - best_size: current best known dominating set size
          - dominated_count: how many vertices are currently dominated
          - graph: the underlying Graph object
          - dominated: boolean list of length n, indicating which vertices are dominated
        Return True if we should prune this branch, False otherwise.
        """
        pass


class SimpleBound(BoundingStrategy):
    def should_prune(self, current_set_size, best_size, dominated_count,
                     graph, dominated):
        # Simple bounding rule: if we've used as many dominators as the best known solution, prune.
        if current_set_size >= best_size:
            return True
        # Otherwise, continue.
        return False


class ImprovedBound(BoundingStrategy):
    def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):
        """
        A more aggressive but still exact bounding strategy.
        """
        # Basic bound: if current set is already worse than best, prune
        if current_set_size >= best_size:
            return True
            
        # Count remaining undominated vertices
        remaining_undominated = graph.n - dominated_count
        
        # If remaining vertices are all dominated, we're done with this branch
        if remaining_undominated == 0:
            return False
            
        # Lower bound calculation: each remaining undominated vertex needs at least
        # one vertex to dominate it (either itself or a neighbor)
        # This is admissible (never overestimates) so it preserves exactness
        remaining_vertices_needed = 0
        
        # Create a set of undominated vertices for faster lookup
        undominated = set(i for i in range(graph.n) if not dominated[i])
        
        # Keep track of which vertices we've considered
        covered = set()
        
        # While there are still undominated vertices
        while undominated and (current_set_size + remaining_vertices_needed < best_size):
            # Find vertex that dominates the most uncovered vertices
            best_v = -1
            best_coverage = 0
            
            for v in range(graph.n):
                if v in covered:
                    continue
                    
                # Calculate how many undominated vertices this v would cover
                coverage = 1 if v in undominated else 0
                coverage += len(undominated.intersection(graph.neighbors_of(v)))
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_v = v
            
            # If we couldn't find a vertex to add, break
            if best_v == -1 or best_coverage == 0:
                break
                
            # Update our tracking sets
            covered.add(best_v)
            remaining_vertices_needed += 1
            
            # Remove vertices that would be dominated by best_v
            if best_v in undominated:
                undominated.remove(best_v)
            undominated -= graph.neighbors_of(best_v)
        
        # If our lower bound exceeds best_size, we can prune
        return current_set_size + remaining_vertices_needed >= best_size


class BranchAndBoundDominatingSetSolver:
    def __init__(self, graph, bounding_strategy: BoundingStrategy, time_limit=1800):
        self.graph = graph
        self.bounding_strategy = bounding_strategy
        self.n = graph.n

        # Global best solution tracking
        self.best_solution = list(range(self.n))  # trivial: all vertices
        self.best_size = self.n

        # 'dominated[v]' indicates whether v is currently dominated
        self.dominated = [False] * self.n
        self.time_limit = time_limit
        self.start_time = None
    def solve(self):
        """
        Solve the Dominating Set problem using branch and bound
        with the chosen bounding strategy. Returns a list of dominators (0-based).
        """
        # Kick off the recursion
        self.start_time = time.time()

        self._branch(0, [], 0)

        # Return the best solution found
        self.best_solution.sort()
        return self.best_solution

    def _branch(self, start_index, current_set, count_dominated):
        """
        Recursive function to perform the branch and bound.

        :param start_index: Next index to consider for branching.
        :param current_set: List of vertices currently chosen in the DS.
        :param count_dominated: Number of vertices dominated so far.
        """
        if time.time() - self.start_time > self.time_limit:
            logger.log("Time limit reached, stopping search.", level=logging.WARNING)
            return
        logger.log(f"Checking if all vertices are dominated [count_dominated={count_dominated}, total={self.n}]", level=logging.WARNING)
        # If all vertices are dominated, we can update the best solution
        if count_dominated == self.n:
            logger.log("All vertices are dominated.", level=logging.WARNING)
            if len(current_set) < self.best_size:
                logger.log(f"New best solution found with size {len(current_set)} (old size was {self.best_size}).", level=logging.WARNING)
                self.best_size = len(current_set)
                self.best_solution = current_set[:]
            return

        logger.log(f"Evaluating bounding strategy [current_set_size={len(current_set)}, best_size={self.best_size}].", level=logging.WARNING)
        # Ask the bounding strategy if we should prune
        if self.bounding_strategy.should_prune(
            current_set_size=len(current_set),
            best_size=self.best_size,
            dominated_count=count_dominated,
            graph=self.graph,
            dominated=self.dominated
        ):
            logger.log("Pruning branch based on bounding strategy.", level=logging.WARNING)
            return

        # If we've reached or passed the last vertex, stop
        if start_index >= self.n:
            logger.log("Reached end of vertices, returning.")
            return

        # Find next undominated vertex (>= start_index)
        next_undominated = -1
        for i in range(start_index, self.n):
            if not self.dominated[i]:
                next_undominated = i
                break

        # If none found, all are dominated => update best solution if needed
        if next_undominated == -1:
            logger.log("All vertices are dominated upon checking again.",level=logging.WARNING)
            if len(current_set) < self.best_size:
                logger.log(f"New best solution found with size {len(current_set)} (old size was {self.best_size}).", level=logging.WARNING)
                self.best_size = len(current_set)
                self.best_solution = current_set[:]
            return

        v = next_undominated
        logger.log(f"Next undominated vertex is {v}. Branching...", level=logging.WARNING)
        # ----- BRANCH 1: Add 'v' to the set -----
        logger.log(f"BRANCH 1: Adding vertex {v} to set {current_set}.", level=logging.WARNING)
        old_dominated = []
        to_dominate = [v] + list(self.graph.neighbors_of(v))

        for w in to_dominate:
            if not self.dominated[w]:
                self.dominated[w] = True
                old_dominated.append(w)

        current_set.append(v)
        logger.log(f"Recursively branching after adding vertex {v}.", level=logging.WARNING)
        self._branch(v+1, current_set, count_dominated + len(old_dominated))

        # revert changes
        logger.log(f"Reverting branch 1 changes for vertex {v}." , level=logging.WARNING)
        current_set.pop()
        for w in old_dominated:
            self.dominated[w] = False

        # ----- BRANCH 2: Do NOT add 'v'; we must ensure 'v' is dominated by a neighbor. -----
        logger.log(f"BRANCH 2: Not adding vertex {v}, trying neighbors.", level=logging.WARNING)
        # Try each neighbor w of v in turn
        for w in self.graph.neighbors_of(v):
            logger.log(f"Attempting to dominate {v} with neighbor {w}.", level=logging.WARNING)
            old_dominated = []
            to_dominate = [w] + list(self.graph.neighbors_of(w))

            for x in to_dominate:
                if not self.dominated[x]:
                    self.dominated[x] = True
                    old_dominated.append(x)

            current_set.append(w)
            logger.log(f"Recursively branching after adding neighbor {w}.", level=logging.WARNING)
            self._branch(v+1, current_set, count_dominated + len(old_dominated))
            # revert
            logger.log(f"Reverting branch 2 changes for neighbor {w}.", level=logging.WARNING)
            current_set.pop()
            for x in old_dominated:
                self.dominated[x] = False


class OptimizedBranchAndBoundDominatingSetSolver:
    def __init__(self, graph, bounding_strategy: BoundingStrategy, time_limit=1800):
        self.graph = graph
        self.bounding_strategy = bounding_strategy
        self.n = graph.n

        # Global best solution tracking
        self.best_solution = list(range(self.n))  # trivial: all vertices
        self.best_size = self.n

        # 'dominated[v]' indicates whether v is currently dominated
        self.dominated = [False] * self.n
        self.time_limit = time_limit
        self.start_time = None
        
        # Internal state for tracking which vertices are in our current solution
        self.in_solution = [False] * self.n
        
        # Track which vertices are required in the solution due to preprocessing
        self.must_include = set()
        
        # Optional: degree and neighbor information cache
        self.degree = [len(graph.neighbors_of(v)) for v in range(self.n)]
        
        # Preprocessing to find vertices that must be in any optimal solution
        self._preprocess()

    def _preprocess(self):
        """
        Apply preprocessing rules to identify vertices that must be in
        any optimal dominating set. These rules preserve exactness.
        """
        # Rule 1: Vertices with degree-1 neighbors where the neighbor has no other neighbors
        # must be in any optimal solution (otherwise we'd need to include all their neighbors)
        for v in range(self.n):
            for w in self.graph.neighbors_of(v):
                if self.degree[w] == 1:  # w only has v as a neighbor
                    self.must_include.add(v)
                    break
        
        # Add all must-include vertices to our initial solution
        for v in self.must_include:
            self._add_to_solution(v)
        
        logger.log(f"Preprocessing found {len(self.must_include)} vertices that must be in the solution: {self.must_include}", level=logging.INFO)
    
    def _add_to_solution(self, v):
        """
        Add vertex v to our solution and update dominated status.
        """
        if self.in_solution[v]:
            return  # Already in solution
            
        self.in_solution[v] = True
        
        # Mark v and its neighbors as dominated
        if not self.dominated[v]:
            self.dominated[v] = True
        
        for w in self.graph.neighbors_of(v):
            if not self.dominated[w]:
                self.dominated[w] = True

    def _remove_from_solution(self, v):
        """
        Remove vertex v from our solution and update dominated status.
        Requires a full recalculation of dominated status.
        """
        if not self.in_solution[v]:
            return  # Not in solution
            
        self.in_solution[v] = False
        
        # Recalculate dominated status from scratch for correctness
        self.dominated = [False] * self.n
        for u in range(self.n):
            if self.in_solution[u]:
                self.dominated[u] = True
                for w in self.graph.neighbors_of(u):
                    self.dominated[w] = True

    def solve(self):
        """
        Solve the Dominating Set problem using branch and bound
        with the chosen bounding strategy. Returns a list of dominators (0-based).
        """
        # Start timing
        self.start_time = time.time()
        
        # Set up initial state for the root of our branch and bound tree
        current_set = []
        dominated_count = sum(self.dominated)
        
        # Add any vertices that must be included from preprocessing
        for v in self.must_include:
            if v not in current_set:
                current_set.append(v)
        
        logger.log(f"Starting branch and bound with {dominated_count} vertices already dominated", level=logging.INFO)
        
        # Start the branch and bound recursion with the first undominated vertex
        self._branch(0, current_set, dominated_count)
        
        # Sort the solution for consistency
        self.best_solution.sort()
        return self.best_solution

    def _find_next_undominated(self, start_index):
        """
        Find the next undominated vertex with lowest index >= start_index.
        """
        for i in range(start_index, self.n):
            if not self.dominated[i]:
                return i
        return -1

    def _branch(self, start_index, current_set, count_dominated):
        """
        Recursive function to perform the branch and bound.
        
        This implementation uses a more efficient approach by:
        1. Prioritizing vertices with high potential impact
        2. Using stronger bounding functions
        3. Avoiding redundant dominated status calculations
        4. Only logging at appropriate levels
        
        :param start_index: Next index to consider for branching.
        :param current_set: List of vertices currently chosen in the DS.
        :param count_dominated: Number of vertices dominated so far.
        """
        # Check time limit with minimal overhead
        if time.time() - self.start_time > self.time_limit:
            logger.log("Time limit reached, stopping search.", level=logging.INFO)
            return
            
        # If all vertices are dominated, we have a valid dominating set
        if count_dominated == self.n:
            # Update best solution if current is better
            if len(current_set) < self.best_size:
                logger.log(f"New best solution found with size {len(current_set)} (old size was {self.best_size}).", level=logging.INFO)
                self.best_size = len(current_set)
                self.best_solution = current_set.copy()
            return
            
        # Use our bounding strategy to decide if we should prune this branch
        if self.bounding_strategy.should_prune(
            current_set_size=len(current_set),
            best_size=self.best_size,
            dominated_count=count_dominated,
            graph=self.graph,
            dominated=self.dominated
        ):
            return
            
        # Find the next undominated vertex
        next_undominated = self._find_next_undominated(start_index)
        
        # If none found, and we haven't achieved full domination,
        # this branch won't lead to a solution
        if next_undominated == -1:
            return
            
        v = next_undominated
        
        # ----- BRANCH 1: Add v to the dominating set -----
        # First, track which vertices become newly dominated
        old_dominated = []
        to_dominate = [v] + list(self.graph.neighbors_of(v))
        
        # Mark them as dominated
        for w in to_dominate:
            if not self.dominated[w]:
                self.dominated[w] = True
                old_dominated.append(w)
                
        # Add v to our current set
        current_set.append(v)
        
        # Recurse with the updated state
        self._branch(v + 1, current_set, count_dominated + len(old_dominated))
        
        # Revert changes for backtracking
        current_set.pop()
        for w in old_dominated:
            self.dominated[w] = False
            
        # If v must be in any optimal solution (from preprocessing), 
        # we can skip the second branch
        if v in self.must_include:
            return
            
        # ----- BRANCH 2: Don't add v; try each neighbor -----
        for w in self.graph.neighbors_of(v):
            # Skip if w is already in our set
            if w in current_set:
                continue
                
            # Similar procedure for adding w instead
            old_dominated = []
            to_dominate = [w] + list(self.graph.neighbors_of(w))
            
            for x in to_dominate:
                if not self.dominated[x]:
                    self.dominated[x] = True
                    old_dominated.append(x)
                    
            current_set.append(w)
            self._branch(v + 1, current_set, count_dominated + len(old_dominated))
            
            # Revert for backtracking
            current_set.pop()
            for x in old_dominated:
                self.dominated[x] = False


class ORToolsDominatingSetSolver:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.n = graph.n
        self.model = cp_model.CpModel()
        self.x_vars = []

    def build_model(self):
        """
        Build the CP-SAT model for the Dominating Set problem:
          Minimize sum(x[v]) subject to: for each vertex i,
          x[i] + sum(x[j] for j in neighbors(i)) >= 1
        """
        # Create Boolean (0-1) decision variables x[v]
        self.x_vars = [self.model.NewBoolVar(f'x_{v}') for v in range(self.n)]

        # Add domination constraints
        for i in range(self.n):
            # x[i] + sum(x[j] for j in neighbors_of(i)) >= 1
            neighbor_vars = [self.x_vars[j] for j in self.graph.neighbors_of(i)]
            self.model.Add(self.x_vars[i] + sum(neighbor_vars) >= 1)

        # Objective: minimize sum(x[v])
        self.model.Minimize(sum(self.x_vars))

    def solve(self, time_limit=None):
        """
        Solve the model, optionally with a time limit (in seconds).
        Returns a list of chosen vertices (0-based) forming a minimum DS if found.
        """
        solver = cp_model.CpSolver()

        # Optionally set a time limit if desired
        if time_limit is not None:
            solver.parameters.max_time_in_seconds = time_limit

        # Solve
        status = solver.Solve(self.model)

        # If a feasible or optimal solution was found, retrieve it
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            chosen = [v for v in range(self.n) if solver.Value(self.x_vars[v]) == 1]
            return chosen
        else:
            # No feasible solution found (should be rare for DS)
            return []


def parse_pace_input(filePath:str):
    """
    Parses a PACE-style Dominating Set input from stdin.
    Format:
      c (comment lines)
      p ds n m
      u v    (m lines of edges)
    Returns:
      n (int): number of vertices
      edges (list of tuples): list of undirected edges (1-based).
    """
    n = 0
    m = 0
    edges = []
    with open(filePath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            # logger.log(f"Line: {line}", level=logging.WARNING)
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


def get_test_files():
    return [filePath for filePath in os.listdir(TEST_FILE_DIRECTORY) if filePath.endswith(".gr")]

def get_sol_files():
    return [filePath for filePath in os.listdir(TEST_FILE_DIRECTORY) if filePath.endswith(".sol")]


def is_valid_dominating_set(adjacency_list, candidate_set):
    """
    Check if 'candidate_set' is a valid dominating set for the graph.

    :param adjacency_list: list of sets, adjacency_list[v] is the neighbors of v (0-based).
    :param candidate_set: iterable of vertex indices (0-based) that form the proposed dominating set.
    :return: True if 'candidate_set' is a dominating set, False otherwise.
    """
    n = len(adjacency_list)

    # Convert the candidate to a set if it's not already, for faster membership tests
    dominators = set(candidate_set)

    for v in range(n):
        # Check if 'v' is dominated
        # A vertex v is dominated if v is in D or (v has a neighbor in D).
        if v in dominators:
            continue  # v is dominated by itself
        else:
            # Check neighbors
            neighbors = adjacency_list[v]
            # If none of the neighbors is in the dominators set, then v is not dominated
            if dominators.isdisjoint(neighbors):
                return False

    # If we reach here, every vertex is dominated
    return True


def draw_graph(testFilePath:str,dominating_set:list=None, title:str="Graph",filePath:str="graph.png"):

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        os.system("pip install matplotlib")
        import matplotlib.pyplot as plt
    try:
        import networkx as nx
    except ImportError:
        os.system("pip install networkx")
        import networkx as nx
    dominating_set = [int(x+1) for x in dominating_set]
    n, edges = parse_pace_input(testFilePath)
    G = nx.Graph()
    for u in range(n):
        G.add_node(u)
    for u, v in edges:
        G.add_edge(u, v)
    # remove vertex 0
    G.remove_node(0)
    colorList = ["red" if v in dominating_set else "blue" for v in G.nodes]
    print(f"Color List: {colorList}")
    print(f"Nodes: {G.nodes}")
    print(f"Dominating Set: {dominating_set}")

    # raise Exception("Stop here")

    # add title to the graph
    plt.title(f"Test Case: {title}")
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=0.05, iterations=1500)
    nx.draw(G, pos=pos,with_labels=True, node_color=colorList)

    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists(f"results/{title}"):
        os.makedirs(f"results/{title}")

    plt.savefig(f"{filePath}")


    plt.clf()


def tabu_search_dominating_set(
        adjacency_list,
        max_iterations=1000,
        tabu_tenure=10,
        time_limit=None,  # in seconds, optional
        seed=None
):
    """
    Basic Tabu Search for Dominating Set.
    :param adjacency_list: list[set], adjacency_list[v] are neighbors of v (0-based).
    :param max_iterations: max number of iterations.
    :param tabu_tenure: how many iterations a move or vertex remains tabu.
    :param time_limit: if given, stop search once time exceeds this.
    :param seed: random seed for reproducibility.
    :return: A list of vertices (0-based) forming the best DS found.
    """
    if seed is not None:
        random.seed(seed)

    n = len(adjacency_list)

    # Start from a quick greedy DS
    current_ds = set(greedy_construct(adjacency_list))
    best_ds = set(current_ds)
    best_size = len(best_ds)

    # Tabu list: {vertex: iteration_when_tabu_expires}
    tabu_list = {}

    start_time = time.time()
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Check time limit
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break

        # Remove expired tabus
        expired = [v for v, expiry in tabu_list.items() if iteration >= expiry]
        for v in expired:
            del tabu_list[v]

        # Explore neighborhood
        candidate_moves = []

        # (a) Try removing a vertex if it remains valid
        for v in list(current_ds):
            if v not in tabu_list and is_still_dominating_without(adjacency_list, current_ds, v):
                # new DS size would be len(current_ds) - 1
                candidate_moves.append(("remove", v, len(current_ds) - 1))

        # (b) Try adding a vertex not in DS
        # this won't usually reduce size, but might help in escaping local minima
        for w in range(n):
            if w not in current_ds and w not in tabu_list:
                candidate_moves.append(("add", w, len(current_ds) + 1))

        # (c) Try swapping: remove v in DS, add w not in DS
        for v in list(current_ds):
            if v in tabu_list:
                continue
            # newly uncovered if we remove v
            newly_uncovered = newly_uncovered_by_remove(adjacency_list, current_ds, v)
            for w in range(n):
                if w in current_ds or w in tabu_list:
                    continue
                # if w covers the newly uncovered
                if covers_all(w, adjacency_list, newly_uncovered):
                    candidate_moves.append(("swap", (v, w), len(current_ds)))

        if not candidate_moves:
            break  # no moves => stuck

        # Pick best move (lowest DS size)
        best_move_size = float('inf')
        best_moves = []
        for move_type, data, new_size in candidate_moves:
            if new_size < best_move_size:
                best_move_size = new_size
                best_moves = [(move_type, data, new_size)]
            elif new_size == best_move_size:
                best_moves.append((move_type, data, new_size))

        chosen_move = random.choice(best_moves)
        move_type, data, _ = chosen_move

        # Execute chosen move
        if move_type == "remove":
            v = data
            current_ds.remove(v)
            tabu_list[v] = iteration + tabu_tenure
        elif move_type == "add":
            w = data
            current_ds.add(w)
            tabu_list[w] = iteration + tabu_tenure
        elif move_type == "swap":
            v, w = data
            current_ds.remove(v)
            current_ds.add(w)
            # mark both as tabu
            tabu_list[v] = iteration + tabu_tenure
            tabu_list[w] = iteration + tabu_tenure

        # If valid, check improvement
        if is_valid_dominating_set2(adjacency_list, current_ds):
            if len(current_ds) < best_size:
                best_size = len(current_ds)
                best_ds = set(current_ds)
        else:
            # If invalid solutions are allowed in your approach, handle with a penalty or keep it.
            # For simplicity, revert:
            if move_type == "remove":
                current_ds.add(v)
                del tabu_list[v]
            elif move_type == "add":
                current_ds.remove(w)
                del tabu_list[w]
            elif move_type == "swap":
                current_ds.remove(w)
                current_ds.add(v)
                del tabu_list[v]
                del tabu_list[w]

    return list(best_ds)