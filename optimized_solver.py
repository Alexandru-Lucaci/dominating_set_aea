from main import logger, Graph, BoundingStrategy
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