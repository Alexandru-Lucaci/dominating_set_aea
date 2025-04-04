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
import resource
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
            print(f"[INFO] [{time.strftime("%d-%m %H:%M:%S", time.localtime())}] : {message}")
        elif level == logging.ERROR :
            self.logger.error(message)
        elif level == logging.WARNING and loggingLevel == logging.WARNING:
            print(f"[WARNING] [{time.strftime("%d-%m %H:%M:%S", time.localtime())}] : {message}")
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
    
    def add_edge(self, u, v):
        """
        Add undirected edge (u, v). 
        u, v are assumed to be 0-based indices.
        """
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)
    
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


class SimpleBound(BoundingStrategy):
    def should_prune(self, current_set_size, best_size, dominated_count,
                     graph, dominated):
        # Simple bounding rule: if we've used as many dominators as the best known solution, prune.
        if current_set_size >= best_size:
            return True
        # Otherwise, continue.
        return False


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
    n, edges = parse_pace_input(testFilePath)
    G = nx.Graph()
    for u in range(n):
        G.add_node(u)
    for u, v in edges:
        G.add_edge(u, v)
    # remove vertex 0
    G.remove_node(0)
    colorList = ["red" if v in dominating_set else "blue" for v in G.nodes]
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
        if is_valid_dominating_set(adjacency_list, current_ds):
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


# Auxiliary functions used by Tabu Search:

def greedy_construct(adjacency_list):
    """
    Simple greedy constructor: repeatedly pick the vertex covering the most uncovered vertices.
    """
    n = len(adjacency_list)
    uncovered = set(range(n))
    ds = []
    while uncovered:
        best_vertex, best_cover = None, -1
        for v in range(n):
            if v not in ds:
                covers = 1 if v in uncovered else 0
                covers += len(adjacency_list[v].intersection(uncovered))
                if covers > best_cover:
                    best_cover = covers
                    best_vertex = v
        ds.append(best_vertex)
        if best_vertex in uncovered:
            uncovered.remove(best_vertex)
        for nb in adjacency_list[best_vertex]:
            uncovered.discard(nb)
    return ds


def newly_uncovered_by_remove(adjacency_list, ds_set, v):
    """
    If we remove v from ds_set, which vertices become uncovered?
    Those are v itself and v's neighbors that are not covered by any other dominator in ds_set.
    """
    if v not in ds_set:
        return []
    ds_minus = ds_set - {v}
    newly = []
    check_list = [v] + list(adjacency_list[v])
    for x in check_list:
        if not is_covered_by(x, ds_minus, adjacency_list):
            newly.append(x)
    return newly


def is_covered_by(vertex, ds, adjacency_list):
    """
    Return True if 'vertex' is in ds or has a neighbor in ds.
    """
    if vertex in ds:
        return True
    for d in ds:
        if (vertex == d) or (vertex in adjacency_list[d]):
            return True
    return False


def covers_all(w, adjacency_list, vertices):
    """Check if w covers all in 'vertices'. i.e. for each x in vertices, x == w or x in adjacency_list[w]."""
    for x in vertices:
        if x != w and x not in adjacency_list[w]:
            return False
    return True


def is_still_dominating_without(adjacency_list, ds_set, v):
    """
    Check if ds_set \ {v} is still a valid dominating set.
    """
    ds_minus = ds_set.copy()
    ds_minus.remove(v)
    return is_valid_dominating_set2(adjacency_list, ds_minus)


def is_valid_dominating_set2(adjacency_list, candidate_set):
    """
    Returns True if candidate_set is a valid dominating set.
    """
    dom = set(candidate_set)
    n = len(adjacency_list)
    for v in range(n):
        if v not in dom:
            # check neighbors
            if dom.isdisjoint(adjacency_list[v]):
                return False
    return True


def run_single_case(
        run_index,
        testFile,
        solFile,
        testFileDir,
        timeLimit=None,
        ourSolution=False,
        orTools=False,
        use_tabu=False
):
    """
    Executes a single test case:
      - Parse the graph from testFile (PACE format)
      - Spawns parallel threads for whichever solver flags are True
      - Compares results with the expected solution file
      - Logs and saves results
    """

    testFilePath = os.path.join(testFileDir, testFile)
    solFilePath = os.path.join(testFileDir, solFile)

    logger.log(f"[Run {run_index}] Checking Test File: {testFile} vs. Sol File: {solFile}")
    logger.log(f"[Run {run_index}] Parsing input file: {testFilePath}")

    # 1) Parse the input file -> adjacency_list
    n, edges = parse_pace_input(testFilePath)
    graph = Graph(n)
    for (u, v) in edges:
        graph.add_edge(u - 1, v - 1)

    # 2) Read solution file -> expected size and solution
    with open(solFilePath, "r") as solIn:
        solLines = solIn.readlines()
        solLines = [l.strip() for l in solLines if not l.startswith("c") and not l.startswith("s")]
        nrOfSolution = int(solLines[0])  # expected DS size
        expected_solution = sorted(int(x) for x in solLines[1:])

    # 3) Define local solver functions
    def solve_branch_and_bound():
        bounding_strategy = SimpleBound()
        solver = BranchAndBoundDominatingSetSolver(graph, bounding_strategy)
        solver.time_limit = timeLimit if timeLimit else 1800  # 30 min default
        start_time = time.time()
        sol = solver.solve()  # 0-based
        elapsed = time.time() - start_time
        return sol, elapsed

    def solve_ortools():
        orToolsSolver = ORToolsDominatingSetSolver(graph)
        orToolsSolver.build_model()
        start_time = time.time()
        sol = orToolsSolver.solve(timeLimit)  # 0-based
        elapsed = time.time() - start_time
        return sol, elapsed

    def solve_tabu():
        # max_iterations can be based on your problem size, or a big number
        # time_limit ensures we won't exceed timeLimit anyway
        start_time = time.time()
        sol = tabu_search_dominating_set(
            adjacency_list=graph.adjacency_list,
            max_iterations=999999,  # or any large # so time_limit is the real cap
            tabu_tenure=10,
            time_limit=timeLimit,
        )
        elapsed = time.time() - start_time
        return sol, elapsed

    # We'll store futures in a dictionary
    futures = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as local_executor:
        if ourSolution:
            futures['ourSolution'] = local_executor.submit(solve_branch_and_bound)
        if orTools:
            futures['orTools'] = local_executor.submit(solve_ortools)
        if use_tabu:
            futures['tabu_search'] = local_executor.submit(solve_tabu)

        # Wait for the submitted tasks to finish
        results = {}
        for key, fut in futures.items():
            try:
                results[key] = fut.result()  # (solution, time)
            except Exception as e:
                logger.log(f"[Run {run_index}] Error in {key} solver: {e}", level=logging.ERROR)
                results[key] = None

    # Now we have up to three solutions in results dict
    # e.g. results['ourSolution'] = (sol, elapsed_time)

    # 4) Validate each solution
    for solver_key, data in results.items():
        if data is None:
            continue  # error occurred
        sol, elapsed = data
        logger.log(f"[Run {run_index}] {solver_key} solution found in {elapsed:.2f}s -> {sol}")
        if not is_valid_dominating_set(graph.adjacency_list, sol):
            raise ValueError(f"[Run {run_index}] {solver_key} solution is invalid for {testFile}!")
        # Check size vs. expected
        if len(sol) != nrOfSolution:
            logger.log(f"[Run {run_index}] {solver_key} DS size = {len(sol)}, expected {nrOfSolution}")
        else:
            logger.log(f"[Run {run_index}] {solver_key} DS size matches expected: {nrOfSolution}")

        # Convert to 1-based
        sol_1 = [v + 1 for v in sorted(sol)]
        logger.log(f"[Run {run_index}] {solver_key} 1-based solution: {sol_1}")

    logger.log(f"[Run {run_index}] Expected (1-based, sorted): {expected_solution}")

    # 5) Save or log results as needed (CSV, images, etc.)
    # For example:
    # if use_tabu and 'tabu_search' in results and results['tabu_search'] is not None:
    #     sol, elapsed = results['tabu_search']
    #     # Save to csv, draw graph, etc.
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists(f"results/{testFile}"):
        os.makedirs(f"results/{testFile}")
    if not os.path.exists(f"results/{testFile}/orTools"):
        os.makedirs(f"results/{testFile}/orTools")
    if not os.path.exists(f"results/{testFile}/ourSolution"):
        os.makedirs(f"results/{testFile}/ourSolution")
    if not os.path.exists(f"results/{testFile}/tabu_search"):
        os.makedirs(f"results/{testFile}/tabu_search")
    if orTools and 'orTools' in results and results['orTools'] is not None:
        sol, elapsed = results['orTools']
        if not os.path.exists(f"results/{testFile}/orTools/graph-ortools.png"):
            with open(f"results/{testFile}/orTools/data.csv", "w") as solOut:
                writer = csv.writer(solOut)
                writer.writerow(["ID","Time","Solution"])
                writer.writerow(["1",elapsed, sol])
        else:
            with open(f"results/{testFile}/orTools/data.csv", "a") as solOut:
                writer = csv.writer(solOut)
                # get max id
                maxid=0
                with open(f"results/{testFile}/orTools/data.csv", "r") as solIn:
                    reader = csv.reader(solIn)
                    for row in reader:
                        if int(row[0]) > maxid:
                            try:
                                maxid = int(row[0])
                            except:
                                pass
                writer.writerow([maxid+1,elapsed, sol])

    if ourSolution and 'ourSolution' in results and results['ourSolution'] is not None:
        sol, elapsed = results['ourSolution']
        if not os.path.exists(f"results/{testFile}/ourSolution/graph-bb.png"):
            with open(f"results/{testFile}/ourSolution/data.csv", "w") as solOut:
                writer = csv.writer(solOut)
                writer.writerow(["ID","Time","Solution"])
                writer.writerow(["1",elapsed, sol])
        else:
            with open(f"results/{testFile}/ourSolution/data.csv", "a") as solOut:
                writer = csv.writer(solOut)
                # get max id
                maxid=0
                with open(f"results/{testFile}/ourSolution/data.csv", "r") as solIn:
                    reader = csv.reader(solIn)
                    for row in reader:
                        if int(row[0]) > maxid:
                            try:
                                maxid = int(row[0])
                            except:
                                pass
                writer.writerow([maxid+1,elapsed, sol])


    if use_tabu and 'tabu_search' in results and results['tabu_search'] is not None:
        sol, elapsed = results['tabu_search']
        if not os.path.exists(f"results/{testFile}/tabu_search/graph-tabu.png"):
            with open(f"results/{testFile}/tabu_search/data.csv", "w") as solOut:
                writer = csv.writer(solOut)
                writer.writerow(["ID","Time","Solution"])
                writer.writerow(["1",elapsed, sol])
        else:
            with open(f"results/{testFile}/tabu_search/data.csv", "a") as solOut:
                writer = csv.writer(solOut)
                # get max id
                maxid=0
                with open(f"results/{testFile}/tabu_search/data.csv", "r") as solIn:
                    reader = csv.reader(solIn)
                    for row in reader:
                        if int(row[0]) > maxid:
                            try:
                                maxid = int(row[0])
                            except:
                                pass
                writer.writerow([maxid+1,elapsed, sol])


    logger.log(f"[Run {run_index}] Test Case {testFile} completed successfully.\n")
def set_memory_limit(gb):
    limit_bytes = gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

def main(numberOfRuns:int=5,timeLimit:int=1800,ourSolution=False,orTools=False,use_tabu=False):
    num_cores = multiprocessing.cpu_count()
    # try:
    #     logger.log(f"Setting memory limit to {num_cores} GB", level=logging.INFO)
    #     set_memory_limit(8)
    # except Exception as e:
    #     logger.log(f"Error setting memory limit: {e}", level=logging.INFO)
    #     raise
    logger.log(f"Detected {num_cores} CPU cores.", level=logging.INFO)

    testFiles = get_test_files()
    # remove test.gr and test_isolated
    testFiles = [filePath for filePath in testFiles if not filePath.startswith("test") and not filePath.startswith("test_isolated")]
    #  sort the files from the _number
    #  of the test file
    testFiles = sorted(testFiles, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    testFiles.append("test_isolated.gr")
    testFiles.append("test.gr")


    solFiles  = get_sol_files()
    # remove test.gr and test_isolated
    solFiles = [filePath for filePath in solFiles if not filePath.startswith("test") and not filePath.startswith("test_isolated")]
    solFiles  = sorted(solFiles, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    solFiles.append("test_isolated.sol")
    solFiles.append("test.sol")

    logger.log(f"Test files: {testFiles}", level=logging.INFO)
    logger.log(f"Solution files: {solFiles}", level=logging.INFO)

    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_list = []
        for i in range(numberOfRuns):
            for testFile in testFiles:
                for solFile in solFiles:
                    if testFile.replace(".gr","") == solFile.replace(".sol",""):
                        fut = executor.submit(
                            run_single_case,
                            i,
                            testFile,
                            solFile,
                            TEST_FILE_DIRECTORY,
                            timeLimit,
                            ourSolution,
                            orTools,
                            use_tabu
                        )
                        future_list.append(fut)

        for future in concurrent.futures.as_completed(future_list):
            try:
                future.result()  # If run_single_case() raises an error, it will be re-raised here
            except Exception as ex:
                logger.log(f"Error in a test thread: {ex}", level=logging.INFO)
                # You could do additional error handling or re-raise
ourSolution = False
orTools = False
tabu_search = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dominating Set Solver")
    parser.add_argument("--log", type=str, help="Log file name")
    parser.add_argument("--logLevel", type=str, help="Logging Level")
    parser.add_argument("--cleanResults", action="store_true", help="Clean Results Directory")
    parser.add_argument("--numberOfRuns", type=int, help="Number of runs")
    parser.add_argument("--timeLimit", type=int, help="Time Limit")
    parser.add_argument("--ourSolution", action="store_true", help="Use our solution")
    parser.add_argument("--orTools", action="store_true", help="Use OR-Tools solution")
    parser.add_argument("--tabu_search", action="store_true", help="Use Tabu Search solution")
    args = parser.parse_args()

    if args.log:
        logger = Logger(args.log)
    if args.logLevel:
        loggingLevel = getattr(logging, args.logLevel)
    if args.cleanResults:
        if os.path.exists("results"):
            os.system("rm -r results")
    if args.timeLimit:
        timeLimit = args.timeLimit
    else:
        timeLimit = 1800
    if args.numberOfRuns:
        numberOfRuns = args.numberOfRuns
    else:
        numberOfRuns = 5

    if args.ourSolution:
        ourSolution = True
    if args.orTools:
        orTools = True
    if args.tabu_search:
        tabu_search = True
    else:
        tabu_search = True
    logger = Logger(log_file_name)
    logger.log("Starting")

    main(numberOfRuns=numberOfRuns, timeLimit=timeLimit, ourSolution=ourSolution, orTools=orTools, use_tabu=tabu_search)


    # testFiles = get_test_files()
    # testFiles = sorted(testFiles)
    # solFiles = get_sol_files()
    # solFiles = sorted(solFiles)
    # for i in range(numberOfRuns):
    #     for testFile in testFiles:
    #         for solFile in solFiles:
    #             if testFile.replace(".gr", "") == solFile.replace(".sol", ""):
    #                 logger.log(f"Test File: {testFile} Sol File: {solFile}")
    #                 logger.log(f"Running Test Case: {testFile}", level=logging.WARNING)
    #                 testFilePath = os.path.join(TEST_FILE_DIRECTORY, testFile)
    #                 solFilePath = os.path.join(TEST_FILE_DIRECTORY, solFile)
    #                 with open(testFilePath, "r") as f:
    #                     logger.log(f"Opened Test File: {testFile}", level=logging.WARNING)
    #                     logger.log(f"Parsing file: {solFile}", level=logging.WARNING)
    #                     n, edges = parse_pace_input(testFilePath)
    #                     logger.log(f"Creating Graph with {n} vertices", level=logging.WARNING)
    #                     logger.log(f"Edges: {edges}", level=logging.WARNING)
    #                     graph = Graph(n)
    #                     for (u, v) in edges:
    #                         logger.log(f"Adding Edge: {u} {v}")
    #                         graph.add_edge(u - 1, v - 1)
    #                     logger.log(f"Graph Created", level=logging.WARNING)
    #                     bounding_strategy = SimpleBound()
    #                     solver = BranchAndBoundDominatingSetSolver(graph, bounding_strategy)
    #                     logger.log(f"Solving Test Case: {testFile}", level=logging.INFO)
    #                     start_time = time.time()
    #                     solution = solver.solve()
    #                     timeInSeconds = time.time() - start_time
    #                     logger.log(f"Solution Found in {timeInSeconds} seconds", level=logging.INFO)
    #                     logger.log(f"Solving test case {testFile} using ORtools", level=logging.INFO)
    #                     orToolsSolver = ORToolsDominatingSetSolver(graph)
    #                     orToolsSolver.build_model()
    #                     start_time = time.time()
    #                     if args.timeLimit:
    #                         orToolsSolution = orToolsSolver.solve(args.timeLimit)
    #                     else:
    #                         orToolsSolution = orToolsSolver.solve()
    #                     timeInSecondsORTools = time.time() - start_time
    #                     logger.log(f"Solution Found in {timeInSecondsORTools} seconds", level=logging.INFO)
    #
    #                     nrOfSolution:int
    #                     with open(solFilePath, "r") as solFile:
    #                         solLines = solFile.readlines()
    #                         try:
    #                             solLines = [int(l) for l in [line.strip() for line in solLines if not line.startswith("c") and not line.startswith("s")]]
    #                             nrOfSolution = solLines[0]
    #                             solLines = solLines[1:]
    #                         except ValueError:
    #                             logger.log(f"Error in parsing solution file: {solFile}, \n{solLines}")
    #
    #                         solution = sorted(solution)
    #                         solLines = sorted(solLines)
    #                         if len(solution) == nrOfSolution and is_valid_dominating_set(graph.adjacency_list, solution) and is_valid_dominating_set(graph.adjacency_list, orToolsSolution):
    #                             orToolsSolution = [v + 1 for v in orToolsSolution]
    #                             logger.log(f"Test Case: {testFile} Passed")
    #                             solution = [v + 1 for v in solution]
    #                             logger.log(f"Solution: {solution}")
    #                             logger.log(f"Expected Solution: {solLines}")
    #                             draw_graph(testFilePath, solution, title=f"{testFile}")
    #                             logger.log(f"OR Tools Solution: {orToolsSolution}")
    #                             if not os.path.exists("results"):
    #                                 os.makedirs("results")
    #                             if not os.path.exists(f"results/{testFile}"):
    #                                 os.makedirs(f"results/{testFile}")
    #                             if not os.path.exists(f"results/{testFile}/orTools"):
    #                                 os.makedirs(f"results/{testFile}/orTools")
    #                             if not os.path.exists(f"results/{testFile}/ourSolution"):
    #                                 os.makedirs(f"results/{testFile}/ourSolution")
    #                             if not os.path.exists(f"results/{testFile}/orTools/data.csv"):
    #                                 with open(f"results/{testFile}/orTools/data.csv", "w") as f:
    #                                     writer = csv.writer(f)
    #                                     writer.writerow(["ID", "Time", "Solution"])
    #                                     writer.writerow(["1", timeInSecondsORTools, orToolsSolution])
    #                             else:
    #                                 with open(f"results/{testFile}/orTools/data.csv", "a") as f:
    #                                     writer = csv.writer(f)
    #                                     maxId = 0
    #                                     with open(f"results/{testFile}/orTools/data.csv", "r") as f:
    #                                         reader = csv.reader(f)
    #                                         for row in reader:
    #                                             try:
    #                                                 maxId = int(row[0])
    #                                             except ValueError:
    #                                                 pass
    #                                     writer.writerow([maxId + 1, timeInSecondsORTools, orToolsSolution])
    #                             if not os.path.exists(f"results/{testFile}/ourSolution/data.csv"):
    #                                 with open(f"results/{testFile}/ourSolution/data.csv", "w") as f:
    #                                     writer = csv.writer(f)
    #                                     writer.writerow(["ID", "Time", "Solution"])
    #                                     writer.writerow(["1", timeInSeconds, solution])
    #                             else:
    #                                 with open(f"results/{testFile}/ourSolution/data.csv", "a") as f:
    #                                     writer = csv.writer(f)
    #                                     maxId = 0
    #                                     with open(f"results/{testFile}/ourSolution/data.csv", "r") as f:
    #                                         reader = csv.reader(f)
    #                                         for row in reader:
    #                                             try:
    #                                                 maxId = int(row[0])
    #                                             except ValueError:
    #                                                 pass
    #                                     writer.writerow([maxId + 1, timeInSeconds, solution])
    #                         else:
    #                             logger.log("Solution is invalid, vertices are not dominated,")
    #                             logger.log(f"Test Case: {testFile} Failed")
    #                             solution = [v + 1 for v in solution]
    #                             logger.log(f"Solution: {solution}")
    #                             logger.log(f"Expected Solution: {solLines}")
    #                             raise ValueError("Solution is invalid, vertices are not dominated")
