from main import BoundingStrategy, Graph
class ImprovedBound(BoundingStrategy):
    def should_prune(self, current_set_size, best_size, dominated_count, graph, dominated):
        """
        A more aggressive but still exact bounding strategy.
        """
        # Basic bound: if current set is already worse than best, prune
        if current_set_size >= best_size:
            return True
            
        # Count remaining undominated vertices
        remaining_undominated = graph.n - dominated_count
        
        # If remaining vertices are all dominated, we're done with this branch
        if remaining_undominated == 0:
            return False
            
        # Lower bound calculation: each remaining undominated vertex needs at least
        # one vertex to dominate it (either itself or a neighbor)
        # This is admissible (never overestimates) so it preserves exactness
        remaining_vertices_needed = 0
        
        # Create a set of undominated vertices for faster lookup
        undominated = set(i for i in range(graph.n) if not dominated[i])
        
        # Keep track of which vertices we've considered
        covered = set()
        
        # While there are still undominated vertices
        while undominated and (current_set_size + remaining_vertices_needed < best_size):
            # Find vertex that dominates the most uncovered vertices
            best_v = -1
            best_coverage = 0
            
            for v in range(graph.n):
                if v in covered:
                    continue
                    
                # Calculate how many undominated vertices this v would cover
                coverage = 1 if v in undominated else 0
                coverage += len(undominated.intersection(graph.neighbors_of(v)))
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_v = v
            
            # If we couldn't find a vertex to add, break
            if best_v == -1 or best_coverage == 0:
                break
                
            # Update our tracking sets
            covered.add(best_v)
            remaining_vertices_needed += 1
            
            # Remove vertices that would be dominated by best_v
            if best_v in undominated:
                undominated.remove(best_v)
            undominated -= graph.neighbors_of(best_v)
        
        # If our lower bound exceeds best_size, we can prune
        return current_set_size + remaining_vertices_needed >= best_size