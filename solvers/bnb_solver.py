import time
import logging
from collections import deque
import heapq

class BranchAndBoundDominatingSetSolver:
    def __init__(self, graph, bounding_strategy, time_limit=1800):
        self.graph = graph
        self.bounding_strategy = bounding_strategy
        self.n = graph.n
        
        # Global best solution tracking
        self.best_solution = None
        self.best_size = float('inf')
        
        # Tracking dominated vertices
        self.dominated = [False] * self.n
        self.dominated_count = 0
        
        # Precompute useful data structures
        self.closed_neighborhood = []
        self.vertex_degrees = []
        for v in range(self.n):
            closed_neighb = self.graph.neighbors_of(v).copy()
            closed_neighb.add(v)
            self.closed_neighborhood.append(closed_neighb)
            self.vertex_degrees.append(len(self.graph.neighbors_of(v)))
        
        # Preprocessing flags
        self.must_include = set()  # Vertices that must be in any solution
        self.excluded = set()      # Vertices that can be excluded
        
        self.time_limit = time_limit
        self.start_time = None
        self.nodes_explored = 0

    def solve(self):
        """Solve the Dominating Set problem using branch and bound."""
        self.start_time = time.time()
        
        # Preprocessing
        self._preprocess()
        
        # Initial solution using multiple heuristics
        initial_solutions = [
            self._greedy_solution(),
            self._maximal_independent_set_heuristic(),
            self._two_approximation()
        ]
        
        # Choose best initial solution
        for sol in initial_solutions:
            if sol and len(sol) < self.best_size:
                self.best_solution = sol
                self.best_size = len(sol)
        
        # Initialize the search with must-include vertices
        initial_set = list(self.must_include)
        for v in initial_set:
            for u in self.closed_neighborhood[v]:
                if not self.dominated[u]:
                    self.dominated[u] = True
                    self.dominated_count += 1
        
        # Start branch and bound
        self._branch(initial_set, 0)
        
        if self.best_solution is None:
            # If no solution found, return all vertices
            return list(range(self.n))
        
        return sorted(self.best_solution)

    def _preprocess(self):
        """Apply preprocessing rules to reduce the problem size."""
        # Rule 1: Isolated vertices must be included
        for v in range(self.n):
            if v not in self.excluded and len(self.graph.neighbors_of(v)) == 0:
                self.must_include.add(v)
        
        # Rule 2: If vertex u is only dominated by vertex v, then v must be included
        for u in range(self.n):
            if u in self.excluded:
                continue
            potential_dominators = [v for v in self.graph.neighbors_of(u) if v not in self.excluded]
            if len(potential_dominators) == 1:
                self.must_include.add(potential_dominators[0])
        
        # Rule 3: If N[u] ⊆ N[v], we can exclude u (v dominates everything u does)
        for u in range(self.n):
            if u in self.excluded or u in self.must_include:
                continue
            for v in range(self.n):
                if v != u and v not in self.excluded:
                    if self.closed_neighborhood[u].issubset(self.closed_neighborhood[v]):
                        self.excluded.add(u)
                        break

    def _greedy_solution(self):
        """Generate a greedy solution for initial upper bound."""
        solution = list(self.must_include)
        covered = [False] * self.n
        
        # Mark already covered vertices
        for v in solution:
            for u in self.closed_neighborhood[v]:
                covered[u] = True
        
        covered_count = sum(covered)
        
        while covered_count < self.n:
            best_v = -1
            best_new_coverage = 0
            
            for v in range(self.n):
                if v in solution or v in self.excluded:
                    continue
                
                new_coverage = 0
                for u in self.closed_neighborhood[v]:
                    if not covered[u]:
                        new_coverage += 1
                
                if new_coverage > best_new_coverage:
                    best_new_coverage = new_coverage
                    best_v = v
            
            if best_v == -1:
                break
                
            solution.append(best_v)
            for u in self.closed_neighborhood[best_v]:
                if not covered[u]:
                    covered[u] = True
                    covered_count += 1
        
        return solution

    def _maximal_independent_set_heuristic(self):
        """Use maximal independent set to get a dominating set."""
        independent_set = []
        used = [False] * self.n
        
        # Sort vertices by degree (ascending) for better independent set
        vertices = list(range(self.n))
        vertices.sort(key=lambda v: self.vertex_degrees[v])
        
        for v in vertices:
            if not used[v] and v not in self.excluded:
                independent_set.append(v)
                used[v] = True
                for u in self.graph.neighbors_of(v):
                    used[u] = True
        
        # The complement might be a dominating set
        dominating = []
        covered = [False] * self.n
        
        # First add the independent set vertices
        for v in independent_set:
            dominating.append(v)
            for u in self.closed_neighborhood[v]:
                covered[u] = True
        
        # Add vertices to cover any remaining uncovered vertices
        for v in range(self.n):
            if not covered[v]:
                # Find a neighbor to add
                for u in self.graph.neighbors_of(v):
                    if u not in dominating:
                        dominating.append(u)
                        for w in self.closed_neighborhood[u]:
                            covered[w] = True
                        break
        
        return dominating

    def _two_approximation(self):
        """2-approximation algorithm based on maximal matching."""
        solution = list(self.must_include)
        covered = [False] * self.n
        
        # Mark already covered vertices
        for v in solution:
            for u in self.closed_neighborhood[v]:
                covered[u] = True
        
        # Find edges where both endpoints are uncovered
        while True:
            found_edge = False
            for v in range(self.n):
                if covered[v] or v in self.excluded:
                    continue
                for u in self.graph.neighbors_of(v):
                    if not covered[u] and u not in self.excluded:
                        # Add both endpoints
                        solution.extend([v, u])
                        for w in self.closed_neighborhood[v]:
                            covered[w] = True
                        for w in self.closed_neighborhood[u]:
                            covered[w] = True
                        found_edge = True
                        break
                if found_edge:
                    break
            
            if not found_edge:
                # Add remaining uncovered vertices
                for v in range(self.n):
                    if not covered[v] and v not in self.excluded:
                        solution.append(v)
                        for w in self.closed_neighborhood[v]:
                            covered[w] = True
                break
        
        return solution

    def _branch(self, current_set, start_idx):
        """Improved branching with better vertex selection."""
        self.nodes_explored += 1
        
        # Time limit check
        if time.time() - self.start_time > self.time_limit:
            return
        
        # Check if all vertices are dominated
        if self.dominated_count == self.n:
            if len(current_set) < self.best_size:
                self.best_size = len(current_set)
                self.best_solution = current_set.copy()
            return
        
        # Pruning
        if len(current_set) >= self.best_size:
            return
        
        # Apply bounding strategy
        if self.bounding_strategy.should_prune(
            current_set_size=len(current_set),
            best_size=self.best_size,
            dominated_count=self.dominated_count,
            graph=self.graph,
            dominated=self.dominated
        ):
            return
        
        # Find the best vertex to branch on
        best_vertex = self._select_best_branching_vertex(start_idx)
        
        if best_vertex == -1:
            return
        
        # Branch 1: Include best_vertex
        newly_dominated = []
        for v in self.closed_neighborhood[best_vertex]:
            if not self.dominated[v]:
                self.dominated[v] = True
                self.dominated_count += 1
                newly_dominated.append(v)
        
        current_set.append(best_vertex)
        self._branch(current_set, best_vertex + 1)
        
        # Restore state
        current_set.pop()
        for v in newly_dominated:
            self.dominated[v] = False
            self.dominated_count -= 1
        
        # Branch 2: Exclude best_vertex, try its best neighbor
        if not self.dominated[best_vertex]:
            # Find the best neighbor to dominate best_vertex
            best_neighbor = -1
            best_neighbor_score = -1
            
            for neighbor in self.graph.neighbors_of(best_vertex):
                if neighbor in current_set or neighbor < start_idx:
                    continue
                
                # Count how many new vertices this neighbor would dominate
                score = 0
                for v in self.closed_neighborhood[neighbor]:
                    if not self.dominated[v]:
                        score += 1
                
                if score > best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor = neighbor
            
            if best_neighbor != -1:
                # Temporarily mark best_vertex as dominated
                old_dominated = self.dominated[best_vertex]
                if not old_dominated:
                    self.dominated[best_vertex] = True
                    self.dominated_count += 1
                
                # Include best_neighbor
                newly_dominated2 = []
                for v in self.closed_neighborhood[best_neighbor]:
                    if not self.dominated[v]:
                        self.dominated[v] = True
                        self.dominated_count += 1
                        newly_dominated2.append(v)
                
                current_set.append(best_neighbor)
                self._branch(current_set, max(best_vertex + 1, best_neighbor + 1))
                
                # Restore
                current_set.pop()
                for v in newly_dominated2:
                    self.dominated[v] = False
                    self.dominated_count -= 1
                
                if not old_dominated:
                    self.dominated[best_vertex] = False
                    self.dominated_count -= 1

    def _select_best_branching_vertex(self, start_idx):
        """Select vertex with best cost-effectiveness ratio."""
        best_vertex = -1
        best_ratio = -1
        
        for v in range(start_idx, self.n):
            if self.dominated[v] or v in self.excluded:
                continue
            
            # Count vertices that would be newly dominated
            new_dominations = 0
            for u in self.closed_neighborhood[v]:
                if not self.dominated[u]:
                    new_dominations += 1
            
            if new_dominations == 0:
                continue
            
            # Prioritize vertices that must be dominated soon
            urgency = 1.0
            if not self.dominated[v]:
                # Count how many potential dominators this vertex has
                potential_dominators = sum(1 for u in self.graph.neighbors_of(v) 
                                         if not self.dominated[u] and u >= start_idx)
                if potential_dominators <= 2:
                    urgency = 3.0
                elif potential_dominators <= 4:
                    urgency = 2.0
            
            ratio = new_dominations * urgency
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_vertex = v
        
        return best_vertex


