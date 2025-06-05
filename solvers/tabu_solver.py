import time
import random
from typing import Set, List, Tuple, Dict
import heapq

def tabu_search_dominating_set_optimized(
        adjacency_list,
        max_iterations=1000,
        tabu_tenure=10,
        time_limit=None,
        seed=None,
        adaptive_tenure=True,
        diversification_freq=50,
        intensification_freq=100
):
    """
    Optimized Tabu Search for Dominating Set with performance improvements.
    
    Key optimizations:
    1. Cached domination checks
    2. Incremental updates for move evaluation
    3. Aspiration criteria
    4. Adaptive tabu tenure
    5. Diversification and intensification strategies
    6. Efficient data structures
    """
    if seed is not None:
        random.seed(seed)

    n = len(adjacency_list)
    
    # Precompute closed neighborhoods (vertex + neighbors)
    closed_neighborhoods = []
    for v in range(n):
        closed_nb = adjacency_list[v].copy()
        closed_nb.add(v)
        closed_neighborhoods.append(closed_nb)
    
    # Initialize with optimized greedy construction
    current_ds = set(greedy_construct_optimized(adjacency_list, closed_neighborhoods))
    best_ds = current_ds.copy()
    best_size = len(best_ds)
    
    # Tabu structures
    tabu_add = {}  # Vertices tabu for addition
    tabu_remove = {}  # Vertices tabu for removal
    
    # Tracking structures for efficiency
    dominated_by = [set() for _ in range(n)]  # dominated_by[v] = set of vertices that dominate v
    domination_count = [0] * n  # How many vertices dominate each vertex
    
    # Initialize domination tracking
    for v in current_ds:
        for u in closed_neighborhoods[v]:
            dominated_by[u].add(v)
            domination_count[u] += 1
    
    # Statistics for adaptive strategies
    no_improvement_count = 0
    recent_sizes = []
    
    # Frequency counts for intensification
    vertex_in_solution_count = [0] * n
    for v in current_ds:
        vertex_in_solution_count[v] += 1
    
    start_time = time.time()
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Clean expired tabu entries
        tabu_add = {v: exp for v, exp in tabu_add.items() if iteration < exp}
        tabu_remove = {v: exp for v, exp in tabu_remove.items() if iteration < exp}
        
        # Adaptive tabu tenure
        if adaptive_tenure and iteration % 20 == 0:
            if no_improvement_count > 10:
                tabu_tenure = min(tabu_tenure + 2, 25)
            elif no_improvement_count < 3:
                tabu_tenure = max(tabu_tenure - 1, 5)
        
        # Find best move
        best_move = None
        best_move_delta = float('inf')
        
        # Evaluate removal moves
        removal_candidates = []
        for v in current_ds:
            if v in tabu_remove and len(current_ds) - 1 >= best_size:
                continue  # Skip unless aspiration
                
            # Check if v can be removed (all vertices still dominated)
            can_remove = True
            for u in closed_neighborhoods[v]:
                if domination_count[u] == 1:  # Only v dominates u
                    can_remove = False
                    break
            
            if can_remove:
                delta = -1
                # Aspiration criterion
                if len(current_ds) - 1 < best_size or v not in tabu_remove:
                    removal_candidates.append((delta, v, "remove"))
        
        # Evaluate addition moves (only if necessary)
        if not removal_candidates:
            for w in range(n):
                if w in current_ds or (w in tabu_add and len(current_ds) + 1 >= best_size):
                    continue
                
                # Count newly dominated vertices
                newly_dominated = sum(1 for u in closed_neighborhoods[w] 
                                    if domination_count[u] == 0)
                
                if newly_dominated > 0:  # Only add if it helps
                    delta = 1
                    if len(current_ds) + 1 < best_size or w not in tabu_add:
                        heapq.heappush(removal_candidates, (delta, w, "add"))
        
        # Evaluate swap moves (limited to promising swaps)
        if len(removal_candidates) < 5:  # Only if we need more options
            for v in list(current_ds)[:20]:  # Limit search
                if v in tabu_remove:
                    continue
                    
                # Find vertices that would become uncovered if v is removed
                critical_vertices = []
                for u in closed_neighborhoods[v]:
                    if domination_count[u] == 1:
                        critical_vertices.append(u)
                
                if not critical_vertices:
                    continue  # v can be removed, so removal is better
                
                # Find best replacement for v
                for w in range(n):
                    if w in current_ds or w == v or w in tabu_add:
                        continue
                    
                    # Check if w covers all critical vertices
                    if all(u in closed_neighborhoods[w] for u in critical_vertices):
                        delta = 0  # Same size
                        if (v, w) not in tabu_remove or len(current_ds) < best_size:
                            heapq.heappush(removal_candidates, (delta, (v, w), "swap"))
                            break  # One good swap per vertex is enough
        
        # Select best move
        if removal_candidates:
            # Sort by delta (size change)
            removal_candidates.sort(key=lambda x: x[0])
            best_move_delta, move_data, move_type = removal_candidates[0]
            best_move = (move_type, move_data)
        
        if best_move is None:
            # Diversification: restart from different solution
            no_improvement_count += 1
            if no_improvement_count > diversification_freq:
                current_ds = set(greedy_construct_random(adjacency_list, n))
                # Rebuild domination tracking
                dominated_by = [set() for _ in range(n)]
                domination_count = [0] * n
                for v in current_ds:
                    for u in closed_neighborhoods[v]:
                        dominated_by[u].add(v)
                        domination_count[u] += 1
                no_improvement_count = 0
            continue
        
        # Apply move
        move_type, move_data = best_move
        
        if move_type == "remove":
            v = move_data
            current_ds.remove(v)
            # Update domination tracking
            for u in closed_neighborhoods[v]:
                dominated_by[u].remove(v)
                domination_count[u] -= 1
            # Update tabu
            tabu_add[v] = iteration + tabu_tenure
            
        elif move_type == "add":
            w = move_data
            current_ds.add(w)
            # Update domination tracking
            for u in closed_neighborhoods[w]:
                dominated_by[u].add(w)
                domination_count[u] += 1
            # Update tabu
            tabu_remove[w] = iteration + tabu_tenure
            
        elif move_type == "swap":
            v, w = move_data
            current_ds.remove(v)
            current_ds.add(w)
            # Update domination tracking
            for u in closed_neighborhoods[v]:
                dominated_by[u].remove(v)
                domination_count[u] -= 1
            for u in closed_neighborhoods[w]:
                dominated_by[u].add(w)
                domination_count[u] += 1
            # Update tabu
            tabu_add[v] = iteration + tabu_tenure
            tabu_remove[w] = iteration + tabu_tenure
        
        # Update statistics
        for v in current_ds:
            vertex_in_solution_count[v] += 1
        
        # Check if solution improved
        if len(current_ds) < best_size:
            best_size = len(current_ds)
            best_ds = current_ds.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Intensification: focus on frequently used vertices
        if iteration % intensification_freq == 0 and no_improvement_count < 10:
            # Try to build solution with most frequently used vertices
            freq_vertices = sorted(range(n), 
                                 key=lambda x: vertex_in_solution_count[x], 
                                 reverse=True)
            intensified_ds = set()
            covered = [False] * n
            
            for v in freq_vertices[:best_size + 5]:
                if not all(covered):
                    intensified_ds.add(v)
                    for u in closed_neighborhoods[v]:
                        covered[u] = True
            
            # Complete if necessary
            for v in range(n):
                if not covered[v]:
                    # Add neighbor or self
                    for u in adjacency_list[v]:
                        if u not in intensified_ds:
                            intensified_ds.add(u)
                            for w in closed_neighborhoods[u]:
                                covered[w] = True
                            break
                    if not covered[v]:
                        intensified_ds.add(v)
                        covered[v] = True
            
            if len(intensified_ds) < len(current_ds):
                current_ds = intensified_ds
                # Rebuild domination tracking
                dominated_by = [set() for _ in range(n)]
                domination_count = [0] * n
                for v in current_ds:
                    for u in closed_neighborhoods[v]:
                        dominated_by[u].add(v)
                        domination_count[u] += 1
                
                if len(current_ds) < best_size:
                    best_size = len(current_ds)
                    best_ds = current_ds.copy()
    
    return list(best_ds)


def greedy_construct_optimized(adjacency_list, closed_neighborhoods):
    """Optimized greedy construction using precomputed neighborhoods."""
    n = len(adjacency_list)
    uncovered = set(range(n))
    ds = []
    
    while uncovered:
        best_vertex = -1
        best_cover = -1
        
        # Use closed neighborhoods for efficiency
        for v in range(n):
            if v not in ds:
                covers = len(closed_neighborhoods[v].intersection(uncovered))
                if covers > best_cover:
                    best_cover = covers
                    best_vertex = v
        
        if best_vertex == -1:
            break
            
        ds.append(best_vertex)
        uncovered.difference_update(closed_neighborhoods[best_vertex])
    
    return ds


def greedy_construct_random(adjacency_list, n):
    """Randomized greedy construction for diversification."""
    uncovered = set(range(n))
    ds = []
    
    while uncovered:
        # Select from top k vertices by coverage
        candidates = []
        for v in range(n):
            if v not in ds:
                covers = 1 if v in uncovered else 0
                covers += len(adjacency_list[v].intersection(uncovered))
                if covers > 0:
                    candidates.append((covers, v))
        
        if not candidates:
            break
            
        # Sort and select from top candidates
        candidates.sort(reverse=True)
        k = min(5, len(candidates))
        _, selected = candidates[random.randint(0, k-1)]
        
        ds.append(selected)
        uncovered.discard(selected)
        uncovered.difference_update(adjacency_list[selected])
    
    return ds


def is_valid_dominating_set(adjacency_list, candidate_set):
    """Fast validation of dominating set."""
    dom = set(candidate_set)
    n = len(adjacency_list)
    
    for v in range(n):
        if v not in dom and dom.isdisjoint(adjacency_list[v]):
            return False
    return True