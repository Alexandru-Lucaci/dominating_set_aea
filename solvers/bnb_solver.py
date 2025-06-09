import time
import logging
from collections import deque
import heapq
from typing import List, Set, Tuple

from graph import Graph
from strategies.bounding import BoundingStrategy, ImprovedBound, StrongBound
from logger import Logger
from utils.validator import is_valid_dominating_set

logger = Logger("ImprovedBnBSolver")

class BranchAndBoundDominatingSetSolver:
    """Enhanced Branch and Bound solver with multiple heuristics and preprocessing."""

    def __init__(self, graph, bounding_strategy, time_limit=1800):
        self.graph = graph
        self.bounding_strategy = bounding_strategy
        self.n = graph.n


        self.best_solution = None
        self.best_size = float('inf')


        self.closed_neighborhood = []
        self.vertex_degrees = []
        self.open_neighborhood = []

        for v in range(self.n):
            neighbors = self.graph.neighbors_of(v)
            closed_neighb = neighbors.copy()
            closed_neighb.add(v)
            self.closed_neighborhood.append(closed_neighb)
            self.open_neighborhood.append(neighbors)
            self.vertex_degrees.append(len(neighbors))


        self.must_include = set()
        self.excluded = set()
        self.vertex_map = {}

        self.time_limit = time_limit
        self.start_time = None
        self.nodes_explored = 0

    def solve(self):
        """Solve the Dominating Set problem using branch and bound."""
        self.start_time = time.time()


        self._advanced_preprocess()


        initial_solutions = self._get_initial_solutions()


        for sol in initial_solutions:
            if sol and len(sol) < self.best_size:
                if self._is_valid_solution(sol):
                    self.best_solution = sol
                    self.best_size = len(sol)
                    logger.log(f"Initial solution of size {self.best_size} found")


        active_vertices = [v for v in range(self.n)
                          if v not in self.excluded]


        initial_set = list(self.must_include)
        dominated = [False] * self.n
        dominated_count = 0


        for v in initial_set:
            for u in self.closed_neighborhood[v]:
                if not dominated[u]:
                    dominated[u] = True
                    dominated_count += 1


        self._improved_branch(initial_set, dominated, dominated_count, 0, active_vertices)

        if self.best_solution is None:
            return list(range(self.n))

        return sorted(self.best_solution)

    def _advanced_preprocess(self):
        """Apply advanced preprocessing rules."""
        changed = True
        while changed:
            changed = False


            for v in range(self.n):
                if v not in self.excluded and v not in self.must_include:
                    if len(self.graph.neighbors_of(v)) == 0:
                        self.must_include.add(v)
                        changed = True


            for v in range(self.n):
                if v in self.excluded or v in self.must_include:
                    continue

                potential_dominators = []
                for u in self.closed_neighborhood[v]:
                    if u not in self.excluded:
                        potential_dominators.append(u)

                if len(potential_dominators) == 1:
                    self.must_include.add(potential_dominators[0])
                    changed = True


            for u in range(self.n):
                if u in self.excluded or u in self.must_include:
                    continue

                for v in range(self.n):
                    if v != u and v not in self.excluded:

                        if self.closed_neighborhood[u].issubset(self.closed_neighborhood[v]):
                            self.excluded.add(u)
                            changed = True
                            break


            for u in range(self.n):
                if u in self.excluded:
                    continue

                for v in range(u + 1, self.n):
                    if v not in self.excluded:
                        if self.closed_neighborhood[u] == self.closed_neighborhood[v]:

                            self.excluded.add(v)
                            changed = True

    def _get_initial_solutions(self) -> List[List[int]]:
        """Generate multiple initial solutions using different strategies."""
        solutions = []


        sol1 = self._greedy_by_coverage()
        if sol1:
            solutions.append(sol1)


        sol2 = self._greedy_by_degree()
        if sol2:
            solutions.append(sol2)


        sol3 = self._maximal_independent_set()
        if sol3:
            solutions.append(sol3)


        sol4 = self._lp_based_heuristic()
        if sol4:
            solutions.append(sol4)

        return solutions

    def _greedy_by_coverage(self) -> List[int]:
        """Greedy algorithm selecting vertices by maximum new coverage."""
        solution = list(self.must_include)
        covered = [False] * self.n

        for v in solution:
            for u in self.closed_neighborhood[v]:
                covered[u] = True

        while not all(covered):
            best_v = -1
            best_coverage = 0

            for v in range(self.n):
                if v in solution or v in self.excluded:
                    continue

                new_coverage = sum(1 for u in self.closed_neighborhood[v]
                                 if not covered[u])

                if new_coverage > best_coverage:
                    best_coverage = new_coverage
                    best_v = v

            if best_v == -1:
                break

            solution.append(best_v)
            for u in self.closed_neighborhood[best_v]:
                covered[u] = True

        return solution

    def _greedy_by_degree(self) -> List[int]:
        """Greedy algorithm selecting high-degree vertices first."""
        solution = list(self.must_include)
        covered = [False] * self.n

        for v in solution:
            for u in self.closed_neighborhood[v]:
                covered[u] = True


        vertices_by_degree = [(v, self.vertex_degrees[v])
                            for v in range(self.n)
                            if v not in self.excluded]
        vertices_by_degree.sort(key=lambda x: x[1], reverse=True)

        for v, _ in vertices_by_degree:
            if v in solution:
                continue

            if not covered[v]:
                solution.append(v)
                for u in self.closed_neighborhood[v]:
                    covered[u] = True

        return solution

    def _maximal_independent_set(self) -> List[int]:
        """Build dominating set from maximal independent set."""

        independent = []
        used = [False] * self.n


        vertices = list(range(self.n))
        vertices.sort(key=lambda v: self.vertex_degrees[v])

        for v in vertices:
            if not used[v] and v not in self.excluded:
                independent.append(v)
                used[v] = True
                for u in self.graph.neighbors_of(v):
                    used[u] = True


        solution = independent.copy()
        covered = [False] * self.n

        for v in solution:
            for u in self.closed_neighborhood[v]:
                covered[u] = True


        for v in range(self.n):
            if not covered[v] and v not in self.excluded:

                added = False
                for u in self.graph.neighbors_of(v):
                    if u not in solution and u not in self.excluded:
                        solution.append(u)
                        for w in self.closed_neighborhood[u]:
                            covered[w] = True
                        added = True
                        break

                if not added:
                    solution.append(v)
                    for w in self.closed_neighborhood[v]:
                        covered[w] = True

        return solution

    def _lp_based_heuristic(self) -> List[int]:
        """Simplified LP-based heuristic."""

        weights = {}
        for v in range(self.n):
            if v in self.excluded:
                weights[v] = 0
            else:

                weights[v] = 1.0 / len(self.closed_neighborhood[v])

        solution = list(self.must_include)
        covered = [False] * self.n

        for v in solution:
            for u in self.closed_neighborhood[v]:
                covered[u] = True


        while not all(covered):
            best_v = -1
            best_ratio = -1

            for v in range(self.n):
                if v in solution or v in self.excluded:
                    continue

                coverage = sum(1 for u in self.closed_neighborhood[v]
                             if not covered[u])
                if coverage > 0:
                    ratio = coverage / weights[v]
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_v = v

            if best_v == -1:
                break

            solution.append(best_v)
            for u in self.closed_neighborhood[best_v]:
                covered[u] = True

        return solution

    def _improved_branch(self, current_set: List[int], dominated: List[bool],
                        dominated_count: int, depth: int, active_vertices: List[int]):
        """Improved branching with better vertex selection and pruning."""
        self.nodes_explored += 1


        if time.time() - self.start_time > self.time_limit:
            return


        if dominated_count == self.n:
            if len(current_set) < self.best_size:
                self.best_size = len(current_set)
                self.best_solution = current_set.copy()
                logger.log(f"New best solution of size {self.best_size}")
            return


        if len(current_set) >= self.best_size:
            return


        if self.bounding_strategy.should_prune(
            current_set_size=len(current_set),
            best_size=self.best_size,
            dominated_count=dominated_count,
            graph=self.graph,
            dominated=dominated
        ):
            return


        branch_vertex = self._select_branch_vertex(dominated, active_vertices, current_set)

        if branch_vertex == -1:
            return


        new_active = [v for v in active_vertices if v != branch_vertex]


        newly_dominated = []
        for v in self.closed_neighborhood[branch_vertex]:
            if not dominated[v]:
                dominated[v] = True
                newly_dominated.append(v)

        current_set.append(branch_vertex)
        self._improved_branch(current_set, dominated,
                            dominated_count + len(newly_dominated),
                            depth + 1, new_active)


        current_set.pop()
        for v in newly_dominated:
            dominated[v] = False


        if not dominated[branch_vertex]:

            best_neighbors = self._get_best_neighbors(branch_vertex, dominated, current_set)

            for neighbor in best_neighbors[:2]:
                newly_dominated2 = []
                for v in self.closed_neighborhood[neighbor]:
                    if not dominated[v]:
                        dominated[v] = True
                        newly_dominated2.append(v)

                current_set.append(neighbor)
                new_active2 = [v for v in new_active if v != neighbor]

                self._improved_branch(current_set, dominated,
                                    dominated_count + len(newly_dominated2),
                                    depth + 1, new_active2)


                current_set.pop()
                for v in newly_dominated2:
                    dominated[v] = False

    def _select_branch_vertex(self, dominated: List[bool],
                            active_vertices: List[int],
                            current_set: List[int]) -> int:
        """Select best vertex to branch on using multiple criteria."""
        best_vertex = -1
        best_score = -1

        for v in active_vertices:
            if dominated[v] or v in current_set:
                continue


            coverage = sum(1 for u in self.closed_neighborhood[v] if not dominated[u])


            undominated_neighbors = sum(1 for u in self.graph.neighbors_of(v)
                                      if not dominated[u] and u not in current_set)
            urgency = 1.0 / (undominated_neighbors + 1)


            degree_factor = self.vertex_degrees[v] / max(self.vertex_degrees)


            score = coverage * (1 + urgency) * (1 + degree_factor)

            if score > best_score:
                best_score = score
                best_vertex = v

        return best_vertex

    def _get_best_neighbors(self, vertex: int, dominated: List[bool],
                          current_set: List[int]) -> List[int]:
        """Get neighbors sorted by their domination potential."""
        neighbors = []

        for n in self.graph.neighbors_of(vertex):
            if n not in current_set and not dominated[n]:
                coverage = sum(1 for u in self.closed_neighborhood[n]
                             if not dominated[u])
                neighbors.append((n, coverage))

        neighbors.sort(key=lambda x: x[1], reverse=True)
        return [n[0] for n in neighbors]

    def _is_valid_solution(self, solution: List[int]) -> bool:
        """Check if solution is a valid dominating set."""
        dominated = [False] * self.n
        for v in solution:
            for u in self.closed_neighborhood[v]:
                dominated[u] = True
        return all(dominated)