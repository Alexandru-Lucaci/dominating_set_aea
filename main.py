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
            logger.log(f"Line: {line}", level=logging.WARNING)
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



def draw_graph(testFilePath:str,dominating_set:list=None, title:str="Graph",bb_solution:bool=True):

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
    nx.draw(G, with_labels=True, node_color=colorList)

    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists(f"results/{title}"):
        os.makedirs(f"results/{title}")
    if bb_solution:
        if not os.path.exists(f"results/{title}/graph-bb.png"):
            plt.savefig(f"results/{title}/graph-bb.png")
    else:
        if not os.path.exists(f"results/{title}/graph-ortools.png"):
            plt.savefig(f"results/{title}/graph-ortools.png")

#     clear plt
    plt.clf()

def run_single_case(
        run_index,
        testFile,
        solFile,
        testFileDir,

        timeLimit=None
):
    """
    Executes a single test case:
      - Parse the graph from testFile (PACE format)
      - Spawn two threads: one for BranchAndBound, one for ORTools
      - Compare results with solution file
      - Save logs and diagrams
    """
    testFilePath = os.path.join(testFileDir, testFile)
    solFilePath = os.path.join(testFileDir, solFile)

    logger.log(f"[Run {run_index}] Checking Test File: {testFile} vs. Sol File: {solFile}")
    logger.log(f"[Run {run_index}] Parsing input file: {testFilePath}")

    # 1) Parse the input file
    n, edges = parse_pace_input(testFilePath)
    graph = Graph(n)
    for (u, v) in edges:
        graph.add_edge(u - 1, v - 1)

    # 2) Read solution file
    with open(solFilePath, "r") as solIn:
        solLines = solIn.readlines()
        solLines = [l.strip() for l in solLines if not l.startswith("c") and not l.startswith("s")]
        # The first line is the number of vertices in the solution
        # The following lines are the actual solution vertices (1-based)
        nrOfSolution = int(solLines[0])
        expected_solution = sorted(int(x) for x in solLines[1:])

    # 3) Solve using your Simple BranchAndBound in one thread
    #    and OR-Tools in another thread
    def solve_branch_and_bound():
        bounding_strategy = SimpleBound()
        solver = BranchAndBoundDominatingSetSolver(graph, bounding_strategy, time_limit=timeLimit)
        start_time = time.time()

        sol = solver.solve()       # 0-based solution
        elapsed = time.time() - start_time

        return sol, elapsed

    def solve_ortools():
        orToolsSolver = ORToolsDominatingSetSolver(graph)
        orToolsSolver.build_model()
        start_time = time.time()
        sol = orToolsSolver.solve(timeLimit)  # 0-based solution
        elapsed = time.time() - start_time

        return sol, elapsed


    # Create a local ThreadPool for these two tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as local_executor:
        future_bb = local_executor.submit(solve_branch_and_bound)
        future_or = local_executor.submit(solve_ortools)
        # Wait for both
        solution_bb, time_bb = future_bb.result()
        solution_or, time_or = future_or.result()


    logger.log(f"[Run {run_index}][{testFile}] B&B solution found in {time_bb:.2f}s -> {solution_bb}")
    logger.log(f"[Run {run_index}][{testFile}] OR-Tools solution found in {time_or:.2f}s -> {solution_or}")


    if not is_valid_dominating_set(graph.adjacency_list, solution_bb):
        raise ValueError(f"[Run {run_index}] BranchAndBound solution is invalid for {testFile}!")
    if not is_valid_dominating_set(graph.adjacency_list, solution_or):
        raise ValueError(f"[Run {run_index}] OR-Tools solution is invalid for {testFile}!")

    # 5) Compare with expected solution size
    #    The user might only want to check that the size is correct or that it’s dominating.
    #    Many DS problems have multiple correct solutions. So we only check validity + size.
    #    If you absolutely need the solver to match EXACT solution lines, that might fail if multiple solutions exist.
    if len(solution_bb) != nrOfSolution:
        logger.log(f"[Run {run_index}][{testFile}] B&B solution has size {len(solution_bb)}, expected {nrOfSolution}")
    else:
        logger.log(f"[Run {run_index}][{testFile}] B&B solution size matches expected: {nrOfSolution}")

    if len(solution_or) != nrOfSolution:
        logger.log(f"[Run {run_index}][{testFile}] OR-Tools solution has size {len(solution_or)}, expected {nrOfSolution}")
    else:
        logger.log(f"[Run {run_index}][{testFile}] OR-Tools solution size matches expected: {nrOfSolution}")

    # Convert 0-based solver solutions to 1-based for logging
    solution_bb_1 = [v + 1 for v in sorted(solution_bb)]
    solution_or_1 = [v + 1 for v in sorted(solution_or)]

    logger.log(f"[Run {run_index}][{testFile}] BranchAndBound 1-based solution: {solution_bb_1}")
    logger.log(f"[Run {run_index}][{testFile}] OR-Tools       1-based solution: {solution_or_1}")
    logger.log(f"[Run {run_index}][{testFile}] Expected (1-based, sorted): {expected_solution}")


    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists(f"results/{testFile}"):
        os.makedirs(f"results/{testFile}")
    if not os.path.exists(f"results/{testFile}/orTools"):
        os.makedirs(f"results/{testFile}/orTools")
    if not os.path.exists(f"results/{testFile}/ourSolution"):
        os.makedirs(f"results/{testFile}/ourSolution")
    if not os.path.exists(f"results/{testFile}/orTools/data.csv"):
        with open(f"results/{testFile}/orTools/data.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Time", "Solution"])
            writer.writerow(["1", time_or, solution_or])
    else:
        with open(f"results/{testFile}/orTools/data.csv", "a") as f:
            writer = csv.writer(f)
            maxId = 0
            with open(f"results/{testFile}/orTools/data.csv", "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        maxId = int(row[0])
                    except ValueError:
                        pass
            writer.writerow([maxId + 1, time_or, solution_or])

    if not os.path.exists(f"results/{testFile}/ourSolution/data.csv"):
        with open(f"results/{testFile}/ourSolution/data.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["ID","Time", "Solution"])
            writer.writerow(["1",time_bb, solution_bb])
    else:
        with open(f"results/{testFile}/ourSolution/data.csv", "a") as f:
            writer = csv.writer(f)
            maxId = 0
            with open(f"results/{testFile}/ourSolution/data.csv", "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        maxId = int(row[0])
                    except ValueError:
                        pass
            writer.writerow([maxId + 1, time_bb, solution_bb])

    # If everything looks good:
    logger.log(f"[Run {run_index}] Test Case {testFile} completed successfully.\n")
def set_memory_limit(gb):
    limit_bytes = gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

def main(numberOfRuns:int=5,timeLimit:int=1800):
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        future_list = []
        for i in range(numberOfRuns):
            for testFile in testFiles:
                for solFile in solFiles:
                    # Check if they match (remove .gr / .sol)
                    if testFile.replace(".gr", "") == solFile.replace(".sol", ""):
                        future = executor.submit(
                            run_single_case,
                            i,
                            testFile,
                            solFile,
                            TEST_FILE_DIRECTORY,
                            timeLimit  # optional
                        )
                        future_list.append(future)

        for future in concurrent.futures.as_completed(future_list):
            try:
                future.result()  # If run_single_case() raises an error, it will be re-raised here
            except Exception as ex:
                logger.log(f"Error in a test thread: {ex}", level=logging.INFO)
                # You could do additional error handling or re-raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dominating Set Solver")
    parser.add_argument("--log", type=str, help="Log file name")
    parser.add_argument("--logLevel", type=str, help="Logging Level")
    parser.add_argument("--cleanResults", action="store_true", help="Clean Results Directory")
    parser.add_argument("--numberOfRuns", type=int, help="Number of runs")
    parser.add_argument("--timeLimit", type=int, help="Time Limit")
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

    logger = Logger(log_file_name)
    logger.log("Starting")

    main(numberOfRuns=numberOfRuns, timeLimit=timeLimit)
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




    # main()
