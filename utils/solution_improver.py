import time
from logger import Logger

logger = Logger("SolutionImprover")

def improve_solution(graph, initial_solution):
    """
    Post-process a dominating set solution to make it minimal.
    Returns an improved (possibly optimal) solution.
    """
    solution = set(initial_solution)
    improved = True

    while improved:
        improved = False


        for v in list(solution):

            temp_solution = solution - {v}


            is_valid = True
            for u in range(graph.n):
                if u not in temp_solution:

                    has_dominator = False
                    for neighbor in graph.neighbors_of(u):
                        if neighbor in temp_solution:
                            has_dominator = True
                            break
                    if not has_dominator:
                        is_valid = False
                        break

            if is_valid:
                solution = temp_solution
                improved = True
                logger.log(f"Removed vertex {v}, new size: {len(solution)}")

    return list(solution)


def local_search_improvement(graph, initial_solution, time_limit=10):
    """
    Apply local search to improve solution quality.
    """
    best_solution = set(initial_solution)
    best_size = len(best_solution)
    start_time = time.time()


    best_solution = set(improve_solution(graph, list(best_solution)))
    best_size = len(best_solution)

    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        improved = False


        for v1 in list(best_solution):
            if time.time() - start_time > time_limit:
                break

            for v2 in range(graph.n):
                if v2 in best_solution:
                    continue


                new_solution = best_solution - {v1} | {v2}


                if is_valid_dominating_set(graph, new_solution):

                    improved_sol = set(improve_solution(graph, list(new_solution)))

                    if len(improved_sol) < best_size:
                        best_solution = improved_sol
                        best_size = len(best_solution)
                        improved = True
                        logger.log(f"Local search improved to size {best_size}")
                        break

            if improved:
                break


        if not improved and iteration < 5:
            for v1 in list(best_solution)[:20]:
                if time.time() - start_time > time_limit:
                    break

                for v2 in list(best_solution)[20:40]:
                    if v2 == v1:
                        continue

                    for v3 in range(graph.n):
                        if v3 in best_solution:
                            continue

                        for v4 in range(v3 + 1, min(graph.n, v3 + 20)):
                            if v4 in best_solution:
                                continue


                            new_solution = best_solution - {v1, v2} | {v3, v4}

                            if is_valid_dominating_set(graph, new_solution):
                                improved_sol = set(improve_solution(graph, list(new_solution)))

                                if len(improved_sol) < best_size:
                                    best_solution = improved_sol
                                    best_size = len(best_solution)
                                    improved = True
                                    logger.log(f"3-opt improved to size {best_size}")
                                    break

                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        if not improved:
            break

    return list(best_solution)


def is_valid_dominating_set(graph, solution):
    """Check if solution is a valid dominating set."""
    solution_set = set(solution)
    for v in range(graph.n):
        if v not in solution_set:

            has_dominator = False
            for neighbor in graph.neighbors_of(v):
                if neighbor in solution_set:
                    has_dominator = True
                    break
            if not has_dominator:
                return False
    return True


def solve_dominating_set_optimized(graph, time_limit=300, debug=False):
    """
    Complete optimized solver for dominating set problem.
    Combines branch-and-bound with post-processing.
    """
    from solvers.bnb_solver import BranchAndBoundDominatingSetSolver
    from strategies.bounding import StrongBound, ImprovedBound

    start_time = time.time()


    strategies = [
        ("StrongBound", StrongBound()),
        ("ImprovedBound", ImprovedBound()),
    ]

    best_solution = None
    best_size = float('inf')


    time_per_strategy = time_limit / len(strategies) * 0.7
    improvement_time = time_limit * 0.3

    for strategy_name, bounding_strategy in strategies:
        if time.time() - start_time > time_limit * 0.7:
            break

        if debug:
            logger.log(f"Trying strategy: {strategy_name}")

        solver = BranchAndBoundDominatingSetSolver(graph, bounding_strategy)
        solver.time_limit = min(time_per_strategy, time_limit - (time.time() - start_time))

        solution = solver.solve()

        if solution and len(solution) < best_size:
            best_solution = solution
            best_size = len(solution)
            if debug:
                logger.log(f"Strategy {strategy_name} found solution of size {best_size}")

    if best_solution is None:

        best_solution = list(range(graph.n))


    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 0:
        if debug:
            logger.log(f"Improving solution with {remaining_time:.1f}s remaining...")


        best_solution = improve_solution(graph, best_solution)


        if remaining_time > 1:
            best_solution = local_search_improvement(graph, best_solution, remaining_time)

    return best_solution