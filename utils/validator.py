def is_valid_dominating_set(adjacency_list, candidate_set):
    """
    Check if 'candidate_set' is a valid dominating set for the graph.

    :param adjacency_list: list of sets, adjacency_list[v] is the neighbors of v (0-based).
    :param candidate_set: iterable of vertex indices (0-based) that form the proposed dominating set.
    :return: True if 'candidate_set' is a dominating set, False otherwise.
    """
    n = len(adjacency_list)

    # Convert the candidate to a set if it's not already, for faster membership tests
    dominators = set(candidate_set)

    for v in range(n):
        # Check if 'v' is dominated
        # A vertex v is dominated if v is in D or (v has a neighbor in D).
        if v in dominators:
            continue  # v is dominated by itself
        else:
            # Check neighbors
            neighbors = adjacency_list[v]
            # If none of the neighbors is in the dominators set, then v is not dominated
            if dominators.isdisjoint(neighbors):
                return False

    # If we reach here, every vertex is dominated
    return True
