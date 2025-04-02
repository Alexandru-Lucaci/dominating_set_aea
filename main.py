import sys
import os
import time
import logging
from abc import ABC, abstractmethod

TEST_FILE_DIRECTORY = "ds_verifier/Dominating Set Verifier/src/test/resources/testset"
log_file_name = "log.txt"
loggingLevel = logging.WARNING


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
        elif level == logging.ERROR:
            self.logger.error(message)
        elif level == logging.WARNING:
            print(f"[WARNING] [{time.strftime("%d-%m %H:%M:%S", time.localtime())}] : {message}")
            self.logger.warning(message)
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

class SimpleBound(BoundingStrategy):
    def should_prune(self, current_set_size, best_size, dominated_count,
                     graph, dominated):
        # Simple bounding rule: if we've used as many dominators as the best known solution, prune.
        if current_set_size >= best_size:
            return True
        # Otherwise, continue.
        return False


class BranchAndBoundDominatingSetSolver:
    def __init__(self, graph, bounding_strategy: BoundingStrategy):
        self.graph = graph
        self.bounding_strategy = bounding_strategy
        self.n = graph.n
        
        # Global best solution tracking
        self.best_solution = list(range(self.n))  # trivial: all vertices
        self.best_size = self.n

        # 'dominated[v]' indicates whether v is currently dominated
        self.dominated = [False] * self.n

    def solve(self):
        """
        Solve the Dominating Set problem using branch and bound
        with the chosen bounding strategy. Returns a list of dominators (0-based).
        """
        # Kick off the recursion
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
        logger.log(f"Checking if all vertices are dominated [count_dominated={count_dominated}, total={self.n}]", level=loggingLevel)
        # If all vertices are dominated, we can update the best solution
        if count_dominated == self.n:
            logger.log("All vertices are dominated.")
            if len(current_set) < self.best_size:
                logger.log(f"New best solution found with size {len(current_set)} (old size was {self.best_size}).", level=loggingLevel)
                self.best_size = len(current_set)
                self.best_solution = current_set[:]
            return
        
        logger.log(f"Evaluating bounding strategy [current_set_size={len(current_set)}, best_size={self.best_size}].", level=loggingLevel)
        # Ask the bounding strategy if we should prune
        if self.bounding_strategy.should_prune(
            current_set_size=len(current_set),
            best_size=self.best_size,
            dominated_count=count_dominated,
            graph=self.graph,
            dominated=self.dominated
        ):
            logger.log("Pruning branch based on bounding strategy.", level=loggingLevel)
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
            logger.log("All vertices are dominated upon checking again.",level=loggingLevel)
            if len(current_set) < self.best_size:
                logger.log(f"New best solution found with size {len(current_set)} (old size was {self.best_size}).", level=loggingLevel)
                self.best_size = len(current_set)
                self.best_solution = current_set[:]
            return
        
        v = next_undominated
        logger.log(f"Next undominated vertex is {v}. Branching...", level=loggingLevel)
        # ----- BRANCH 1: Add 'v' to the set -----
        logger.log(f"BRANCH 1: Adding vertex {v} to set {current_set}.", level=loggingLevel)
        old_dominated = []
        to_dominate = [v] + list(self.graph.neighbors_of(v))
        
        for w in to_dominate:
            if not self.dominated[w]:
                self.dominated[w] = True
                old_dominated.append(w)
        
        current_set.append(v)
        logger.log(f"Recursively branching after adding vertex {v}.")
        self._branch(v+1, current_set, count_dominated + len(old_dominated))
        
        # revert changes
        logger.log(f"Reverting branch 1 changes for vertex {v}." , level=loggingLevel)
        current_set.pop()
        for w in old_dominated:
            self.dominated[w] = False
        
        # ----- BRANCH 2: Do NOT add 'v'; we must ensure 'v' is dominated by a neighbor. -----
        logger.log(f"BRANCH 2: Not adding vertex {v}, trying neighbors.", level=loggingLevel)
        # Try each neighbor w of v in turn
        for w in self.graph.neighbors_of(v):
            logger.log(f"Attempting to dominate {v} with neighbor {w}.", level=loggingLevel)
            old_dominated = []
            to_dominate = [w] + list(self.graph.neighbors_of(w))
            
            for x in to_dominate:
                if not self.dominated[x]:
                    self.dominated[x] = True
                    old_dominated.append(x)
            
            current_set.append(w)
            logger.log(f"Recursively branching after adding neighbor {w}.", level=loggingLevel)
            self._branch(v+1, current_set, count_dominated + len(old_dominated))
            # revert
            logger.log(f"Reverting branch 2 changes for neighbor {w}.", level=loggingLevel)
            current_set.pop()
            for x in old_dominated:
                self.dominated[x] = False

################################################################
# 4. Main / I/O Handling
################################################################
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
            logger.log(f"Line: {line}")
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


def main():
    # Parse input
    n, edges = parse_pace_input()
    
    # Build the graph (0-based indexing internally)
    graph = Graph(n)
    for (u, v) in edges:
        graph.add_edge(u - 1, v - 1)
    
    # Choose a bounding strategy (for example, the SimpleBound)
    bounding_strategy = SimpleBound()
    
    # Create the solver
    solver = BranchAndBoundDominatingSetSolver(graph, bounding_strategy)
    
    # Solve
    solution = solver.solve()
    
    # Print solution in PACE format
    print(len(solution))
    for v in solution:
        print(v + 1)  # convert back to 1-based

def draw_graph(graph):
    import subprocess
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
    G = nx.Graph()
    for u in range(graph.n):
        G.add_node(u)
    for u in range(graph.n):
        for v in graph.neighbors_of(u):
            G.add_edge(u, v)
    nx.draw(G, with_labels=True)
    plt.show()
logger = Logger(log_file_name)
if __name__ == "__main__":
    logger.log("Starting")
    start_time = time.time()
    testFiles = get_test_files()
    solFiles = get_sol_files()
    testFiles =["test.gr"]
    solFiles = ["test.sol"]
    for testFile in testFiles:
        for solFile in solFiles:
            if testFile.replace(".gr", "") == solFile.replace(".sol", ""):
                logger.log(f"Test File: {testFile} Sol File: {solFile}")
                logger.log(f"Running Test Case: {testFile}", level=loggingLevel)
                testFilePath = os.path.join(TEST_FILE_DIRECTORY, testFile)
                solFilePath = os.path.join(TEST_FILE_DIRECTORY, solFile)
                with open(testFilePath, "r") as f:
                    logger.log(f"Opened Test File: {testFile}", level=loggingLevel)
                    logger.log(f"Parsing file: {solFile}", level=loggingLevel)
                    n, edges = parse_pace_input(testFilePath)
                    logger.log(f"Creating Graph with {n} vertices", level=loggingLevel)
                    logger.log(f"Edges: {edges}", level=loggingLevel)
                    graph = Graph(n)
                    for (u, v) in edges:
                        logger.log(f"Adding Edge: {u} {v}")
                        graph.add_edge(u - 1, v - 1)
                    logger.log(f"Graph Created", level=loggingLevel)
                    draw_graph(graph)
                    bounding_strategy = SimpleBound()
                    solver = BranchAndBoundDominatingSetSolver(graph, bounding_strategy)
                    solution = solver.solve()
                    nrOfSolution:int
                    with open(solFilePath, "r") as solFile:
                        solLines = solFile.readlines()
                        try:
                            solLines = [int(l) for l in [line.strip() for line in solLines if not line.startswith("c") and not line.startswith("s")]]
                            nrOfSolution = solLines[0]
                            solLines = solLines[1:]
                        except ValueError:
                            logger.log(f"Error in parsing solution file: {solFile}, \n{solLines}")

                        if solution == solLines:
                            logger.log(f"Test Case: {testFile} Passed")
                            logger.log(f"Solution: {solution}")
                            logger.log(f"Expected Solution: {solLines}")
                        else:
                            logger.log(f"Test Case: {testFile} Failed")
                            logger.log(f"Solution: {solution}")
                            logger.log(f"Expected Solution: {solLines}")


    # main()
