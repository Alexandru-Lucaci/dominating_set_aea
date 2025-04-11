import sys
import os
import subprocess
import time
import logging
import csv
import argparse
import concurrent.futures
import multiprocessing
import random
import shutil

try:
    import resource
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "resource"])
    # import resource

# Project imports
from logger import Logger
from graph import Graph
from strategies.bounding import SimpleBound
from solvers.ortools_solver import ORToolsDominatingSetSolver
from solvers.bnb_solver import BranchAndBoundDominatingSetSolver
from solvers.tabu_solver import (
    tabu_search_dominating_set,
    is_valid_dominating_set as is_valid_dominating_set2  # To avoid naming conflict
)
from utils.parser import parse_pace_input, get_test_files, get_sol_files
from utils.validator import is_valid_dominating_set
from utils.visualization import draw_graph

# Constants
TEST_FILE_DIRECTORY = "ds_verifier/Dominating Set Verifier/src/test/resources/testset"
log_file_name = "log.txt"
loggingLevel = logging.INFO

# Initialize global logger
logger = Logger(log_file_name)


def set_memory_limit(gb):
    """Set memory limit for the process in GB."""
    limit_bytes = gb * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def run_single_case(
        run_index,
        testFile,
        solFile,
        testFileDir,
        timeLimit=None,
        ourSolution=False,
        orTools=False,
        use_tabu=False
):
    """
    Executes a single test case:
      - Parse the graph from testFile (PACE format)
      - Spawns parallel threads for whichever solver flags are True
      - Compares results with the expected solution file
      - Logs and saves results
    """

    testFilePath = os.path.join(testFileDir, testFile)
    solFilePath = os.path.join(testFileDir, solFile)

    logger.log(f"[Run {run_index}] Checking Test File: {testFile} vs. Sol File: {solFile}")
    logger.log(f"[Run {run_index}] Parsing input file: {testFilePath}")

    # 1) Parse the input file -> adjacency_list
    n, edges = parse_pace_input(testFilePath)
    graph = Graph(n)
    for (u, v) in edges:
        graph.add_edge(u - 1, v - 1)

    # 2) Read solution file -> expected size and solution
    with open(solFilePath, "r") as solIn:
        solLines = solIn.readlines()
        solLines = [l.strip() for l in solLines if not l.startswith("c") and not l.startswith("s")]
        nrOfSolution = int(solLines[0])  # expected DS size
        expected_solution = sorted(int(x) for x in solLines[1:])

    # 3) Define local solver functions
    def solve_branch_and_bound():
        bounding_strategy = SimpleBound()
        solver = BranchAndBoundDominatingSetSolver(graph, bounding_strategy)
        solver.time_limit = timeLimit if timeLimit else 1800  # 30 min default
        start_time = time.time()
        sol = solver.solve()  # 0-based
        solver_key = "ourSolution"
        elapsed = time.time() - start_time

        # Create directories if they don't exist
        ensure_directories_exist(testFile)

        if sol is None:
            logger.log(f"[Run {run_index}] No solution found for {testFile}!", level=logging.WARNING)
            return [], elapsed
        else:
            logger.log(f"[Run {run_index}] {solver_key} solution found in {elapsed:.2f}s -> {sol}")
            if not is_valid_dominating_set(graph.adjacency_list, sol):
                logger.log(f"[Run {run_index}] {solver_key} solution is invalid for {testFile}!")

            # Check size vs. expected
            if len(sol) != nrOfSolution:
                logger.log(f"[Run {run_index}] {solver_key} DS size = {len(sol)}, expected {nrOfSolution}")
            else:
                logger.log(f"[Run {run_index}] {solver_key} DS size matches expected: {nrOfSolution}")

            # Convert to 1-based
            sol_1 = [v + 1 for v in sorted(sol)]
            logger.log(f"[Run {run_index}] {solver_key} 1-based solution: {sol_1}")

            logger.log(f"[Run {run_index}] Expected (1-based, sorted): {expected_solution}")

            logger.log(f"[Run {run_index}] Solution found for {testFile}: {sol}", level=logging.WARNING)

            # Save results to CSV
            save_results_to_csv(testFile, "ourSolution", run_index, elapsed, sol, expected_solution, nrOfSolution)

        return sol, elapsed

    def solve_ortools():
        orToolsSolver = ORToolsDominatingSetSolver(graph)
        orToolsSolver.build_model()
        start_time = time.time()
        sol = orToolsSolver.solve(timeLimit)  # 0-based
        elapsed = time.time() - start_time

        # Create directories if they don't exist
        ensure_directories_exist(testFile)

        if sol is None:
            logger.log(f"[Run {run_index}] No solution found for {testFile}!", level=logging.WARNING)
            return [], elapsed
        else:
            logger.log(f"[Run {run_index}] orTools solution found in {elapsed:.2f}s -> {sol}")
            if not is_valid_dominating_set(graph.adjacency_list, sol):
                logger.log(f"[Run {run_index}] orTools solution is invalid for {testFile}!")

            # Check size vs. expected
            if len(sol) != nrOfSolution:
                logger.log(f"[Run {run_index}] orTools DS size = {len(sol)}, expected {nrOfSolution}")
            else:
                logger.log(f"[Run {run_index}] orTools DS size matches expected: {nrOfSolution}")

            # Convert to 1-based
            sol_1 = [v + 1 for v in sorted(sol)]
            logger.log(f"[Run {run_index}] orTools 1-based solution: {sol_1}")

            logger.log(f"[Run {run_index}] Expected (1-based, sorted): {expected_solution}")

            logger.log(f"[Run {run_index}] Solution found for {testFile}: {sol}", level=logging.WARNING)

            # Save results to CSV
            save_results_to_csv(testFile, "orTools", run_index, elapsed, sol, expected_solution, nrOfSolution)

        return sol, elapsed

    def solve_tabu():
        # max_iterations can be based on your problem size, or a big number
        # time_limit ensures we won't exceed timeLimit anyway
        start_time = time.time()

        actual_time_limit = 480 if timeLimit is None else timeLimit
        sol = tabu_search_dominating_set(
            adjacency_list=graph.adjacency_list,
            max_iterations=900000000,  # or any large # so time_limit is the real cap
            tabu_tenure=25,
            time_limit=actual_time_limit,
        )
        elapsed = time.time() - start_time

        # Create directories if they don't exist
        ensure_directories_exist(testFile)

        if sol is None:
            logger.log(f"[Run {run_index}] No solution found for {testFile}!", level=logging.WARNING)
            return [], elapsed
        else:
            logger.log(f"[Run {run_index}] Tabu Search solution found in {elapsed:.2f}s -> {sol}")
            if not is_valid_dominating_set(graph.adjacency_list, sol):
                logger.log(f"[Run {run_index}] Tabu Search solution is invalid for {testFile}!")

            # Check size vs. expected
            if len(sol) != nrOfSolution:
                logger.log(f"[Run {run_index}] Tabu Search DS size = {len(sol)}, expected {nrOfSolution}")
            else:
                logger.log(f"[Run {run_index}] Tabu Search DS size matches expected: {nrOfSolution}")

            # Convert to 1-based
            sol_1 = [v + 1 for v in sorted(sol)]
            logger.log(f"[Run {run_index}] Tabu Search 1-based solution: {sol_1}")

            logger.log(f"[Run {run_index}] Expected (1-based, sorted): {expected_solution}")

            logger.log(f"[Run {run_index}] Solution found for {testFile}: {sol}", level=logging.WARNING)

            # Save results to CSV
            save_results_to_csv(testFile, "tabu_search", run_index, elapsed, sol, expected_solution, nrOfSolution)

        return sol, elapsed

    def ensure_directories_exist(testFile):
        """Create required directories for results if they don't exist."""
        if not os.path.exists("results"):
            os.makedirs("results")
        if not os.path.exists(f"results/{testFile}"):
            os.makedirs(f"results/{testFile}")
        if not os.path.exists(f"results/{testFile}/orTools"):
            os.makedirs(f"results/{testFile}/orTools")
        if not os.path.exists(f"results/{testFile}/ourSolution"):
            os.makedirs(f"results/{testFile}/ourSolution")
        if not os.path.exists(f"results/{testFile}/tabu_search"):
            os.makedirs(f"results/{testFile}/tabu_search")

    def save_results_to_csv(testFile, solver_type, run_index, elapsed, sol, expected_solution, nrOfSolution):
        """Save solver results to a CSV file."""
        csv_path = f"results/{testFile}/{solver_type}/data.csv"

        # Check if CSV exists and write header or append data
        if not os.path.exists(csv_path):
            with open(csv_path, "w") as solOut:
                writer = csv.writer(solOut)
                writer.writerow(["ID", "Time", "Solution", "Expected Solution", "Number of Vertices",
                                 "Number of vertices expected", "Valid Solution"])
                writer.writerow([
                    "1",
                    elapsed,
                    sol,
                    expected_solution,
                    len(sol),
                    nrOfSolution,
                    is_valid_dominating_set(graph.adjacency_list, sol)
                ])
        else:
            maxid = 0
            with open(csv_path, "r") as solIn:
                reader = csv.reader(solIn)
                for row in reader:
                    try:
                        if int(row[0]) > maxid:
                            maxid = int(row[0])
                    except:
                        pass

            with open(csv_path, "a") as solOut:
                writer = csv.writer(solOut)
                writer.writerow([
                    maxid + 1,
                    elapsed,
                    sol,
                    expected_solution,
                    len(sol),
                    nrOfSolution,
                    is_valid_dominating_set(graph.adjacency_list, sol)
                ])

    futures = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as local_executor:
        if ourSolution:
            futures['ourSolution'] = local_executor.submit(solve_branch_and_bound)
        if orTools:
            futures['orTools'] = local_executor.submit(solve_ortools)
        if use_tabu:
            futures['tabu_search'] = local_executor.submit(solve_tabu)

        # Wait for the submitted tasks to finish
        results = {}
        for key, fut in futures.items():
            try:
                print(f"[Run {run_index}] Waiting for {key} solver to finish...")
                results[key] = fut.result()  # (solution, time)
                print(f"[Run {run_index}] {key} solver finished.")
                print(results[key])

            except Exception as e:
                # logger.log(f"[Run {run_index}] Error in {key} solver: {e}", level=logging.ERROR)
                print(f"[Run {run_index}] Error in {key} solver: {e}")
                # sys.exit(5)
                results[key] = None

    # 4) Validate each solution
    for solver_key, data in results.items():
        if data is None:
            continue  # error occurred
        sol, elapsed = data
        logger.log(f"[Run {run_index}] {solver_key} solution found in {elapsed:.2f}s -> {sol}")
        if not is_valid_dominating_set(graph.adjacency_list, sol):
            logger.log(f"[Run {run_index}] {solver_key} solution is invalid for {testFile}!")
        # Check size vs. expected
        if len(sol) != nrOfSolution:
            logger.log(f"[Run {run_index}] {solver_key} DS size = {len(sol)}, expected {nrOfSolution}")
        else:
            logger.log(f"[Run {run_index}] {solver_key} DS size matches expected: {nrOfSolution}")

        # Convert to 1-based
        sol_1 = [v + 1 for v in sorted(sol)]
        logger.log(f"[Run {run_index}] {solver_key} 1-based solution: {sol_1}")

    logger.log(f"[Run {run_index}] Expected (1-based, sorted): {expected_solution}")
    logger.log(f"[Run {run_index}] Test Case {testFile} completed successfully.\n")


def main(numberOfRuns=5, timeLimit=1800, ourSolution=False, orTools=False, use_tabu=False):
    """
    Main function to run the Dominating Set solvers on all test cases.

    Args:
        numberOfRuns: Number of times to run each test case
        timeLimit: Time limit in seconds for each solver
        ourSolution: Whether to use the branch and bound solver
        orTools: Whether to use the OR-Tools solver
        use_tabu: Whether to use the Tabu Search solver
    """
    num_cores = multiprocessing.cpu_count()
    # try:
    #     logger.log(f"Setting memory limit to {num_cores} GB", level=logging.INFO)
    #     set_memory_limit(8)
    # except Exception as e:
    #     logger.log(f"Error setting memory limit: {e}", level=logging.INFO)
    #     raise
    logger.log(f"Detected {num_cores} CPU cores.", level=logging.INFO)

    testFiles = get_test_files(TEST_FILE_DIRECTORY)
    # remove test.gr and test_isolated
    testFiles = [filePath for filePath in testFiles if
                 not filePath.startswith("test") and not filePath.startswith("test_isolated")]
    #  sort the files from the _number
    #  of the test file
    testFiles = sorted(testFiles, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    testFiles.append("test_isolated.gr")
    testFiles.append("test.gr")

    solFiles = get_sol_files(TEST_FILE_DIRECTORY)
    # remove test.gr and test_isolated
    solFiles = [filePath for filePath in solFiles if
                not filePath.startswith("test") and not filePath.startswith("test_isolated")]
    solFiles = sorted(solFiles, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    solFiles.append("test_isolated.sol")
    solFiles.append("test.sol")

    logger.log(f"Test files: {testFiles}", level=logging.INFO)
    logger.log(f"Solution files: {solFiles}", level=logging.INFO)

    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_list = []
        for i in range(numberOfRuns):
            for testFile in testFiles:
                for solFile in solFiles:
                    if testFile.replace(".gr", "") == solFile.replace(".sol", ""):
                        fut = executor.submit(
                            run_single_case,
                            i,
                            testFile,
                            solFile,
                            TEST_FILE_DIRECTORY,
                            timeLimit,
                            ourSolution,
                            orTools,
                            use_tabu
                        )
                        future_list.append(fut)

        for future in concurrent.futures.as_completed(future_list):
            try:
                future.result()  # If run_single_case() raises an error, it will be re-raised here
            except Exception as ex:
                # logger.log(f"Error in a test thread: {ex}", level=logging.INFO)
                print(f"Error in a test thread: {ex}")
                # You could do additional error handling or re-raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dominating Set Solver")
    parser.add_argument("--log", type=str, help="Log file name")
    parser.add_argument("--logLevel", type=str, help="Logging Level")
    parser.add_argument("--cleanResults", action="store_true", help="Clean Results Directory")
    parser.add_argument("--numberOfRuns", type=int, help="Number of runs")
    parser.add_argument("--timeLimit", type=int, help="Time Limit")
    parser.add_argument("--ourSolution", action="store_true", help="Use our solution")
    parser.add_argument("--orTools", action="store_true", help="Use OR-Tools solution")
    parser.add_argument("--tabu_search", action="store_true", help="Use Tabu Search solution")
    args = parser.parse_args()

    if args.log:
        logger = Logger(args.log)
    if args.logLevel:
        loggingLevel = getattr(logging, args.logLevel)
    if args.cleanResults:
        if os.path.exists("results"):
            print("Removing results directory")
            #  remove all results folder
            shutil.rmtree("results")

    # Set default parameters
    timeLimit = args.timeLimit if args.timeLimit else 1800
    numberOfRuns = args.numberOfRuns if args.numberOfRuns else 5
    ourSolution = args.ourSolution
    orTools = args.orTools
    tabu_search = args.tabu_search

    # If no solver is selected, use all
    if not (ourSolution or orTools or tabu_search):
        ourSolution = True
        orTools = True
        tabu_search = True

    logger.log("Starting Dominating Set Solver")
    main(numberOfRuns=numberOfRuns, timeLimit=timeLimit,
         ourSolution=ourSolution, orTools=orTools, use_tabu=tabu_search)
    logger.log("Finished executing all tests")