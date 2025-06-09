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

        if current_set_size >= best_size:
            return True


        undominated_count = graph.n - dominated_count

        if undominated_count == 0:
            return False


        max_degree = 0
        undominated_vertices = []

        for v in range(graph.n):
            if not dominated[v]:
                undominated_vertices.append(v)
                degree = len(graph.neighbors_of(v))
                if degree > max_degree:
                    max_degree = degree


        max_coverage = min(max_degree + 1, undominated_count)


        min_additional = (undominated_count + max_coverage - 1) // max_coverage

        if current_set_size + min_additional >= best_size:
            return True


        if len(undominated_vertices) <= 10:

            independent = True
            for v in undominated_vertices:
                for u in undominated_vertices:
                    if v != u and u in graph.neighbors_of(v):
                        independent = False
                        break
                if not independent:
                    break

            if independent:

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

        return False


class EfficientDegreeBound(BoundingStrategy):
    """
    A more efficient degree-based bounding strategy.
    """

    def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):

        if current_set_size >= best_size:
            return True


        undominated_vertices = [v for v in range(graph.n) if not dominated[v]]
        undominated_count = len(undominated_vertices)


        if undominated_count == 0:
            return False


        if current_set_size + math.ceil(undominated_count / 5) > best_size:

            return True


        if len(undominated_vertices) < 50:

            max_coverage = 0
            for v in range(graph.n):
                if dominated[v]:
                    continue


                coverage = 0
                covered = {v}
                covered.update(graph.neighbors_of(v))

                for u in undominated_vertices:
                    if u in covered:
                        coverage += 1

                max_coverage = max(max_coverage, coverage)


                if coverage > undominated_count / 2:
                    break


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

        if current_set_size >= best_size:
            return True


        undominated_count = graph.n - dominated_count


        if graph.n < 100:


            min_required = math.ceil(undominated_count / 5)
            if current_set_size + min_required > best_size:
                return True
        elif graph.n < 500:

            min_required = math.ceil(undominated_count / 10)
            if current_set_size + min_required > best_size:
                return True
        else:

            min_required = math.ceil(undominated_count / 15)
            if current_set_size + min_required > best_size:
                return True


        if 50 < graph.n < 500 and undominated_count < graph.n / 3:


            components = self._count_components(graph, dominated)
            if current_set_size + components > best_size:
                return True

        return False

    def _count_components(self, graph, dominated):
        """Count connected components of undominated vertices."""

        visited = dominated.copy()
        components = 0

        for v in range(len(dominated)):
            if visited[v]:
                continue


            components += 1


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

        if current_set_size >= best_size:
            return True


        if best_size == float('inf'):
            return False



        undominated_count = graph.n - dominated_count
        min_additional = math.ceil(undominated_count / 6.0)

        if current_set_size + min_additional > best_size:
            return True

        return False
class ImprovedBound(BoundingStrategy):
    """More aggressive bounding strategy with multiple pruning rules."""

    def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):

        if current_set_size >= best_size:
            return True

        undominated_count = graph.n - dominated_count
        if undominated_count == 0:
            return False


        lower_bound = self._calculate_lower_bound(graph, dominated, undominated_count)

        if current_set_size + lower_bound >= best_size:
            return True

        return False

    def _calculate_lower_bound(self, graph, dominated, undominated_count):
        """Calculate a lower bound on additional vertices needed."""

        undominated = []
        max_degree_among_undominated = 0

        for v in range(graph.n):
            if not dominated[v]:
                undominated.append(v)

                degree = 1
                for u in graph.neighbors_of(v):
                    if not dominated[u]:
                        degree += 1
                max_degree_among_undominated = max(max_degree_among_undominated, degree)

        if max_degree_among_undominated == 0:
            return undominated_count


        basic_bound = (undominated_count + max_degree_among_undominated - 1) // max_degree_among_undominated


        if len(undominated) <= 20:

            independent_vertices = 0
            for v in undominated:
                is_independent = True
                for u in undominated:
                    if v != u and u in graph.neighbors_of(v):
                        is_independent = False
                        break
                if is_independent:
                    independent_vertices += 1


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


        independent_count = 0
        used = set()

        for v in undominated:
            if v not in used:
                independent_count += 1
                used.add(v)

                for u in undominated:
                    if u != v and u in graph.neighbors_of(v):
                        used.add(u)

        return independent_count

    def _matching_based_bound(self, graph, dominated):
        """Lower bound based on matching."""
        undominated = [v for v in range(graph.n) if not dominated[v]]
        if not undominated:
            return 0


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


        isolated = len(undominated) - len(matched)
        return matching_size + isolated

    def _degree_based_bound(self, graph, dominated):
        """Improved degree-based bound."""
        undominated = [v for v in range(graph.n) if not dominated[v]]
        if not undominated:
            return 0


        max_coverage = 0
        for v in range(graph.n):
            if not dominated[v]:
                coverage = sum(1 for u in graph.closed_neighborhood[v]
                             if not dominated[u])
                max_coverage = max(max_coverage, coverage)

        if max_coverage == 0:
            return len(undominated)

        return (len(undominated) + max_coverage - 1) // max_coverage

class StrongBound(BoundingStrategy):
    """Stronger bounding strategy with multiple lower bound techniques."""

    def __init__(self):

        self._closed_neighborhoods = {}

    def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):
        if current_set_size >= best_size:
            return True

        undominated_count = graph.n - dominated_count
        if undominated_count == 0:
            return False


        if graph.n not in self._closed_neighborhoods:
            self._closed_neighborhoods[graph.n] = []
            for v in range(graph.n):
                closed_neighb = graph.neighbors_of(v).copy()
                closed_neighb.add(v)
                self._closed_neighborhoods[graph.n].append(closed_neighb)


        graph.closed_neighborhood = self._closed_neighborhoods[graph.n]


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


        independent_count = 0
        used = set()

        for v in undominated:
            if v not in used:
                independent_count += 1
                used.add(v)

                for u in undominated:
                    if u != v and u in graph.neighbors_of(v):
                        used.add(u)

        return independent_count

    def _matching_based_bound(self, graph, dominated):
        """Lower bound based on matching."""
        undominated = [v for v in range(graph.n) if not dominated[v]]
        if not undominated:
            return 0


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


        isolated = len(undominated) - len(matched)
        return matching_size + isolated

    def _degree_based_bound(self, graph, dominated):
        """Improved degree-based bound."""
        undominated = [v for v in range(graph.n) if not dominated[v]]
        if not undominated:
            return 0


        max_coverage = 0
        for v in range(graph.n):
            if not dominated[v]:
                coverage = sum(1 for u in graph.closed_neighborhood[v]
                             if not dominated[u])
                max_coverage = max(max_coverage, coverage)

        if max_coverage == 0:
            return len(undominated)

        return (len(undominated) + max_coverage - 1) // max_coverage
