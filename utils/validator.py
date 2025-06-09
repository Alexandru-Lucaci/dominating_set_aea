def is_valid_dominating_set(adjacency_list, candidate_set):
    """
    Check if 'candidate_set' is a valid dominating set for the graph.

    :param adjacency_list: list of sets, adjacency_list[v] is the neighbors of v (0-based).
    :param candidate_set: iterable of vertex indices (0-based) that form the proposed dominating set.
    :return: True if 'candidate_set' is a dominating set, False otherwise.
    """
    n = len(adjacency_list)


    dominators = set(candidate_set)

    for v in range(n):


        if v in dominators:
            continue
        else:

            neighbors = adjacency_list[v]

            if dominators.isdisjoint(neighbors):
                return False


    return True
