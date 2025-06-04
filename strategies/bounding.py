from abc import ABC, abstractmethod
import math
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
    """Simplified and more aggressive bounding strategy."""
    
    def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):
        # Basic pruning
        if current_set_size >= best_size:
            return True
        
        # Calculate remaining vertices to dominate
        undominated_count = graph.n - dominated_count
        
        if undominated_count == 0:
            return False
        
        # Optimistic bound: assume each new vertex can dominate at most max_degree vertices
        max_degree = 0
        undominated_vertices = []
        
        for v in range(graph.n):
            if not dominated[v]:
                undominated_vertices.append(v)
                degree = len(graph.neighbors_of(v))
                if degree > max_degree:
                    max_degree = degree
        
        # Each vertex can dominate itself and its neighbors
        max_coverage = min(max_degree + 1, undominated_count)
        
        # Lower bound on additional vertices needed
        min_additional = (undominated_count + max_coverage - 1) // max_coverage
        
        if current_set_size + min_additional >= best_size:
            return True
        
        # Additional pruning for sparse regions
        if len(undominated_vertices) <= 10:  # Only for small subproblems
            # Check if undominated vertices form independent set
            independent = True
            for v in undominated_vertices:
                for u in undominated_vertices:
                    if v != u and u in graph.neighbors_of(v):
                        independent = False
                        break
                if not independent:
                    break
            
            if independent:
                # Each vertex needs its own dominator
                if current_set_size + len(undominated_vertices) >= best_size:
                    return True
        
        return False
class FastBound(BoundingStrategy):
    """
    A faster and simplified bounding strategy that focuses on efficiency.
    """

    def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):

        if current_set_size >= best_size:
            return True
        if best_size == float('inf'):
            return False
        undominated_count = graph.n - dominated_count
        total_edges = sum(len(graph.neighbors_of(v)) for v in range(graph.n))
        avg_degree = total_edges / graph.n if graph.n > 0 else 0

        avg_domination = 1 + avg_degree
        conservative_factor = max(3.0, avg_domination / 2)
        estimated_additional = max(1, math.ceil(undominated_count / conservative_factor))


        if current_set_size + estimated_additional > best_size:
            return True

        return False  # Continue searching


class EfficientDegreeBound(BoundingStrategy):
    """
    A more efficient degree-based bounding strategy.
    """

    def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):
        # Basic check
        if current_set_size >= best_size:
            return True

        # Find undominated vertices
        undominated_vertices = [v for v in range(graph.n) if not dominated[v]]
        undominated_count = len(undominated_vertices)

        # If no undominated vertices, we have a complete solution
        if undominated_count == 0:
            return False

        # Quick check with simple bound
        if current_set_size + math.ceil(undominated_count / 5) > best_size:
            # Assume very optimistically that each new vertex covers 5 others
            return True

        # Only do more complex calculations if we're still unsure
        if len(undominated_vertices) < 50:  # Only do for smaller sets
            # Find the vertex with maximum coverage
            max_coverage = 0
            for v in range(graph.n):
                if dominated[v]:
                    continue  # Skip dominated vertices

                # Count undominated vertices this would cover
                coverage = 0
                covered = {v}  # Covers itself
                covered.update(graph.neighbors_of(v))

                for u in undominated_vertices:
                    if u in covered:
                        coverage += 1

                max_coverage = max(max_coverage, coverage)

                # Early exit if we find a vertex with very high coverage
                if coverage > undominated_count / 2:
                    break

            # If max_coverage is valid, calculate lower bound
            if max_coverage > 0:
                min_additional = math.ceil(undominated_count / max_coverage)
                if current_set_size + min_additional > best_size:
                    return True

        return False


class DynamicBound(BoundingStrategy):
    """
    A dynamic bounding strategy that adapts based on graph size and search progress.
    """

    def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):
        # Basic bound
        if current_set_size >= best_size:
            return True

        # Remaining vertices to cover
        undominated_count = graph.n - dominated_count

        # Dynamic approach based on graph size
        if graph.n < 100:
            # For small graphs, use more aggressive bounds
            # Estimate we need at least 1 vertex for every 5 undominated
            min_required = math.ceil(undominated_count / 5)
            if current_set_size + min_required > best_size:
                return True
        elif graph.n < 500:
            # For medium graphs, be more conservative
            min_required = math.ceil(undominated_count / 10)
            if current_set_size + min_required > best_size:
                return True
        else:
            # For large graphs, use very conservative bounds to avoid excessive pruning
            min_required = math.ceil(undominated_count / 15)
            if current_set_size + min_required > best_size:
                return True

        # Additional check for medium-sized graphs: connected components analysis
        if 50 < graph.n < 500 and undominated_count < graph.n / 3:
            # Count connected components of undominated vertices
            # This is expensive so only do it when we have fewer undominated vertices
            components = self._count_components(graph, dominated)
            if current_set_size + components > best_size:
                return True

        return False

    def _count_components(self, graph, dominated):
        """Count connected components of undominated vertices."""
        # Simple approximation: at minimum we need one vertex per isolated component
        visited = dominated.copy()  # Start with dominated vertices marked as visited
        components = 0

        for v in range(len(dominated)):
            if visited[v]:
                continue

            # Found a new component
            components += 1

            # BFS to mark all vertices in this component
            queue = [v]
            visited[v] = True

            while queue:
                current = queue.pop(0)
                for neighbor in graph.neighbors_of(current):
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

        return components


class ConservativeBound(BoundingStrategy):
    """
    A simplified, very conservative bounding strategy.
    This should rarely prune incorrectly but may be less efficient.
    """

    def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):
        # Basic pruning: if current solution already exceeds or equals best, prune
        if current_set_size >= best_size:
            return True

        # Don't prune if we haven't found a solution yet
        if best_size == float('inf'):
            return False

        # Very conservative estimate: 1 vertex can cover at most 6 vertices on average
        # (itself + up to 5 neighbors)
        undominated_count = graph.n - dominated_count
        min_additional = math.ceil(undominated_count / 6.0)

        if current_set_size + min_additional > best_size:
            return True

        return False
class ImprovedBound(BoundingStrategy):
    """More aggressive bounding strategy with multiple pruning rules."""
    
    def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):
        # Basic pruning
        if current_set_size >= best_size:
            return True
        
        undominated_count = graph.n - dominated_count
        if undominated_count == 0:
            return False
        
        # Calculate a tighter lower bound
        lower_bound = self._calculate_lower_bound(graph, dominated, undominated_count)
        
        if current_set_size + lower_bound >= best_size:
            return True
        
        return False
    
    def _calculate_lower_bound(self, graph, dominated, undominated_count):
        """Calculate a lower bound on additional vertices needed."""
        # Find undominated vertices
        undominated = []
        max_degree_among_undominated = 0
        
        for v in range(graph.n):
            if not dominated[v]:
                undominated.append(v)
                # Count how many other undominated vertices this can dominate
                degree = 1  # itself
                for u in graph.neighbors_of(v):
                    if not dominated[u]:
                        degree += 1
                max_degree_among_undominated = max(max_degree_among_undominated, degree)
        
        if max_degree_among_undominated == 0:
            return undominated_count
        
        # Basic lower bound
        basic_bound = (undominated_count + max_degree_among_undominated - 1) // max_degree_among_undominated
        
        # Check for independent sets among undominated vertices
        if len(undominated) <= 20:  # Only for small sets
            # Quick independence check
            independent_vertices = 0
            for v in undominated:
                is_independent = True
                for u in undominated:
                    if v != u and u in graph.neighbors_of(v):
                        is_independent = False
                        break
                if is_independent:
                    independent_vertices += 1
            
            # Each independent vertex needs its own dominator
            basic_bound = max(basic_bound, independent_vertices)
        
        return basic_bound
    
class StrongBound(BoundingStrategy):
    """Stronger bounding strategy with multiple lower bound techniques."""
    
    def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):
        if current_set_size >= best_size:
            return True
        
        undominated_count = graph.n - dominated_count
        if undominated_count == 0:
            return False
        
        # Calculate multiple lower bounds and take maximum
        lb1 = self._clique_based_bound(graph, dominated)
        lb2 = self._matching_based_bound(graph, dominated)
        lb3 = self._degree_based_bound(graph, dominated)
        
        lower_bound = max(lb1, lb2, lb3)
        
        return current_set_size + lower_bound >= best_size
    
    def _clique_based_bound(self, graph, dominated):
        """Lower bound based on independent sets."""
        undominated = [v for v in range(graph.n) if not dominated[v]]
        if not undominated:
            return 0
        
        # Find size of maximal independent set among undominated
        independent_count = 0
        used = set()
        
        for v in undominated:
            if v not in used:
                independent_count += 1
                used.add(v)
                # Mark neighbors as used
                for u in undominated:
                    if u != v and u in graph.neighbors_of(v):
                        used.add(u)
        
        return independent_count
    
    def _matching_based_bound(self, graph, dominated):
        """Lower bound based on matching."""
        undominated = [v for v in range(graph.n) if not dominated[v]]
        if not undominated:
            return 0
        
        # Find maximal matching among undominated vertices
        matched = set()
        matching_size = 0
        
        for v in undominated:
            if v not in matched:
                for u in undominated:
                    if u != v and u not in matched and u in graph.neighbors_of(v):
                        matched.add(v)
                        matched.add(u)
                        matching_size += 1
                        break
        
        # Each edge in matching needs at least one dominator
        isolated = len(undominated) - len(matched)
        return matching_size + isolated
    
    def _degree_based_bound(self, graph, dominated):
        """Improved degree-based bound."""
        undominated = [v for v in range(graph.n) if not dominated[v]]
        if not undominated:
            return 0
        
        # Find maximum possible coverage
        max_coverage = 0
        for v in range(graph.n):
            if not dominated[v]:
                coverage = sum(1 for u in graph.closed_neighborhood[v] 
                             if not dominated[u])
                max_coverage = max(max_coverage, coverage)
        
        if max_coverage == 0:
            return len(undominated)
        
        return (len(undominated) + max_coverage - 1) // max_coverage