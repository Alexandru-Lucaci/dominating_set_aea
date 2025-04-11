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
    """
     Adaptive bounding strategy that uses different bounds based on
     graph structure and search progress.
     """

    def __init__(self):
        self.dynamic = DynamicBound()
        self.efficient = EfficientDegreeBound()
        self.fast = FastBound()
        self._graph_stats = {}
        # Cached values to avoid recomputation
        self._max_vertex_coverage = {}  # Mapping of graph.n -> coverage_value

    def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):
        # Always do the simplest check first
        if current_set_size >= best_size:
            # print(f"AdaptiveBound basic pruning: current_size={current_set_size} >= best_size={best_size}")
            return True

        # Don't prune if we haven't found a solution yet
        if best_size == float('inf'):
            # print("AdaptiveBound: No solution found yet, not pruning")
            return False

        # Check graph characteristics if not cached
        if graph.n not in self._graph_stats:
            # print(f"AdaptiveBound: Analyzing graph with {graph.n} vertices")
            self._graph_stats[graph.n] = self._analyze_graph(graph)

        graph_stats = self._graph_stats[graph.n]
        graph_density = graph_stats['density']
        # print(f"AdaptiveBound: Graph density = {graph_density:.4f}, avg_degree = {graph_stats['avg_degree']:.2f}")

        # Stage 1: Very conservative bound for sparse graphs
        if graph_density < 0.05:  # Very sparse
            # In sparse graphs, each vertex dominates fewer others
            # Use a more conservative estimate
            undominated_count = graph.n - dominated_count
            min_additional = math.ceil(undominated_count / 3)  # Assume 1 vertex covers 3 on average
            # print(f"AdaptiveBound (sparse graph): undominated={undominated_count}, min_additional={min_additional}")
            if current_set_size + min_additional > best_size:
                # print(f"AdaptiveBound pruning (sparse): {current_set_size}+{min_additional} > {best_size}")
                return True

        # Stage 2: Adapt based on search progress
        progress = dominated_count / graph.n
        # print(f"AdaptiveBound: Search progress = {progress:.2f} ({dominated_count}/{graph.n} vertices dominated)")

        # For dense graphs or early in search, use simple bounds
        if graph_density > 0.3 or progress < 0.3:
            # Very simple check with conservative estimate
            undominated_count = graph.n - dominated_count
            min_additional = math.ceil(undominated_count / 5)  # 1 vertex covers 5 at most
            # print(f"AdaptiveBound (dense/early): undominated={undominated_count}, min_additional={min_additional}")
            if current_set_size + min_additional > best_size:
                # print(f"AdaptiveBound pruning (dense/early): {current_set_size}+{min_additional} > {best_size}")
                return True
            return False

        # Mid search, try efficient bound
        if progress < 0.7:
            # print("AdaptiveBound: Using EfficientDegreeBound (mid-search)")
            return self.efficient.should_prune(current_set_size, best_size, dominated_count, graph, dominated)

        # Late in search, apply dynamic bound
        # print("AdaptiveBound: Using DynamicBound (late-search)")
        return self.dynamic.should_prune(current_set_size, best_size, dominated_count, graph, dominated)

    def _analyze_graph(self, graph):
        """Analyze graph properties for adaptive decisions."""
        n = graph.n
        if n <= 1:
            return {'density': 0.0, 'avg_degree': 0.0}

        total_edges = sum(len(graph.neighbors_of(v)) for v in range(n))
        avg_degree = total_edges / n

        # Graph density = |E| / (|V| * (|V|-1)/2)
        max_possible_edges = n * (n - 1) / 2
        density = total_edges / max_possible_edges if max_possible_edges > 0 else 0

        return {
            'density': density,
            'avg_degree': avg_degree
        }

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
