import time
import logging

from graph import Graph
from strategies.bounding import BoundingStrategy
from logger import Logger

# Get global logger
# This is a placeholder - you'll need to import the actual logger instance
logger = None  # This will be set by the main application
logger = Logger("BranchAndBoundSolver")
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
            logger.log("All vertices are dominated upon checking again.", level=logging.WARNING)
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
        logger.log(f"Reverting branch 1 changes for vertex {v}.", level=logging.WARNING)
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
