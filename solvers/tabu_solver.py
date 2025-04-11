import time
import random

def tabu_search_dominating_set(
        adjacency_list,
        max_iterations=1000,
        tabu_tenure=10,
        time_limit=None,  # in seconds, optional
        seed=None
):
    """
    Basic Tabu Search for Dominating Set.
    :param adjacency_list: list[set], adjacency_list[v] are neighbors of v (0-based).
    :param max_iterations: max number of iterations.
    :param tabu_tenure: how many iterations a move or vertex remains tabu.
    :param time_limit: if given, stop search once time exceeds this.
    :param seed: random seed for reproducibility.
    :return: A list of vertices (0-based) forming the best DS found.
    """
    if seed is not None:
        random.seed(seed)

    n = len(adjacency_list)

    # Start from a quick greedy DS
    current_ds = set(greedy_construct(adjacency_list))
    best_ds = set(current_ds)
    best_size = len(best_ds)

    # Tabu list: {vertex: iteration_when_tabu_expires}
    tabu_list = {}

    start_time = time.time()
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Check time limit
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break

        # Remove expired tabus
        expired = [v for v, expiry in tabu_list.items() if iteration >= expiry]
        for v in expired:
            del tabu_list[v]

        # Explore neighborhood
        candidate_moves = []

        # (a) Try removing a vertex if it remains valid
        for v in list(current_ds):
            if v not in tabu_list and is_still_dominating_without(adjacency_list, current_ds, v):
                # new DS size would be len(current_ds) - 1
                candidate_moves.append(("remove", v, len(current_ds) - 1))

        # (b) Try adding a vertex not in DS
        # this won't usually reduce size, but might help in escaping local minima
        for w in range(n):
            if w not in current_ds and w not in tabu_list:
                candidate_moves.append(("add", w, len(current_ds) + 1))

        # (c) Try swapping: remove v in DS, add w not in DS
        for v in list(current_ds):
            if v in tabu_list:
                continue
            # newly uncovered if we remove v
            newly_uncovered = newly_uncovered_by_remove(adjacency_list, current_ds, v)
            for w in range(n):
                if w in current_ds or w in tabu_list:
                    continue
                # if w covers the newly uncovered
                if covers_all(w, adjacency_list, newly_uncovered):
                    candidate_moves.append(("swap", (v, w), len(current_ds)))

        if not candidate_moves:
            break  # no moves => stuck

        # Pick best move (lowest DS size)
        best_move_size = float('inf')
        best_moves = []
        for move_type, data, new_size in candidate_moves:
            if new_size < best_move_size:
                best_move_size = new_size
                best_moves = [(move_type, data, new_size)]
            elif new_size == best_move_size:
                best_moves.append((move_type, data, new_size))

        chosen_move = random.choice(best_moves)
        move_type, data, _ = chosen_move

        # Execute chosen move
        if move_type == "remove":
            v = data
            current_ds.remove(v)
            tabu_list[v] = iteration + tabu_tenure
        elif move_type == "add":
            w = data
            current_ds.add(w)
            tabu_list[w] = iteration + tabu_tenure
        elif move_type == "swap":
            v, w = data
            current_ds.remove(v)
            current_ds.add(w)
            # mark both as tabu
            tabu_list[v] = iteration + tabu_tenure
            tabu_list[w] = iteration + tabu_tenure

        # If valid, check improvement
        if is_valid_dominating_set(adjacency_list, current_ds):
            if len(current_ds) < best_size:
                best_size = len(current_ds)
                best_ds = set(current_ds)
        else:
            # If invalid solutions are allowed in your approach, handle with a penalty or keep it.
            # For simplicity, revert:
            if move_type == "remove":
                current_ds.add(v)
                del tabu_list[v]
            elif move_type == "add":
                current_ds.remove(w)
                del tabu_list[w]
            elif move_type == "swap":
                current_ds.remove(w)
                current_ds.add(v)
                del tabu_list[v]
                del tabu_list[w]

    return list(best_ds)


# Auxiliary functions used by Tabu Search:

def greedy_construct(adjacency_list):
    """
    Simple greedy constructor: repeatedly pick the vertex covering the most uncovered vertices.
    """
    n = len(adjacency_list)
    uncovered = set(range(n))
    ds = []
    while uncovered:
        best_vertex, best_cover = None, -1
        for v in range(n):
            if v not in ds:
                covers = 1 if v in uncovered else 0
                covers += len(adjacency_list[v].intersection(uncovered))
                if covers > best_cover:
                    best_cover = covers
                    best_vertex = v
        ds.append(best_vertex)
        if best_vertex in uncovered:
            uncovered.remove(best_vertex)
        for nb in adjacency_list[best_vertex]:
            uncovered.discard(nb)
    return ds


def newly_uncovered_by_remove(adjacency_list, ds_set, v):
    """
    If we remove v from ds_set, which vertices become uncovered?
    Those are v itself and v's neighbors that are not covered by any other dominator in ds_set.
    """
    if v not in ds_set:
        return []
    ds_minus = ds_set - {v}
    newly = []
    check_list = [v] + list(adjacency_list[v])
    for x in check_list:
        if not is_covered_by(x, ds_minus, adjacency_list):
            newly.append(x)
    return newly


def is_covered_by(vertex, ds, adjacency_list):
    """
    Return True if 'vertex' is in ds or has a neighbor in ds.
    """
    if vertex in ds:
        return True
    for d in ds:
        if (vertex == d) or (vertex in adjacency_list[d]):
            return True
    return False


def covers_all(w, adjacency_list, vertices):
    """Check if w covers all in 'vertices'. i.e. for each x in vertices, x == w or x in adjacency_list[w]."""
    for x in vertices:
        if x != w and x not in adjacency_list[w]:
            return False
    return True


def is_still_dominating_without(adjacency_list, ds_set, v):
    """
    Check if ds_set \ {v} is still a valid dominating set.
    """
    ds_minus = ds_set.copy()
    ds_minus.remove(v)
    return is_valid_dominating_set(adjacency_list, ds_minus)


def is_valid_dominating_set(adjacency_list, candidate_set):
    """
    Returns True if candidate_set is a valid dominating set.
    """
    dom = set(candidate_set)
    n = len(adjacency_list)
    for v in range(n):
        if v not in dom:
            # check neighbors
            if dom.isdisjoint(adjacency_list[v]):
                return False
    return True
