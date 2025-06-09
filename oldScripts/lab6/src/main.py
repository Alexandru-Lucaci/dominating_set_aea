import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import time

from docplex.mp.model import Model


def draw_graph_v2(graph, dominating_set, filename):
    """Draws and saves a graph visualization with highlighted dominating set nodes."""
    G = nx.Graph(graph)  # Create graph from adjacency list

    num_nodes = len(G.nodes)  # Get number of nodes
    plt.figure(figsize=(min(15, num_nodes // 10 + 5), min(15, num_nodes // 10 + 5)))  # Scale figure size

    # Choose a layout dynamically based on graph size
    if num_nodes > 200:
        pos = nx.kamada_kawai_layout(G)  # Best for large graphs
    elif num_nodes > 50:
        pos = nx.spring_layout(G, k=3 / (num_nodes ** 0.5), seed=42)  # Adjust k to spread nodes
    else:
        pos = nx.spring_layout(G, seed=42)  # Default for small graphs

    # Node sizes and font sizes
    node_size = max(50, 800 - num_nodes * 2)  # Reduce size for large graphs
    font_size = max(6, 12 - num_nodes // 50)  # Reduce font for large graphs

    # Draw all nodes
    nx.draw(G, pos, with_labels=True, node_color="lightgray", edge_color="gray",
            node_size=node_size, font_size=font_size, alpha=0.9)

    # Highlight dominating set nodes in red
    nx.draw_networkx_nodes(G, pos, nodelist=dominating_set, node_color="red", node_size=node_size + 100)

    # Save the figure
    plt.savefig(filename, format="png", bbox_inches="tight")
    plt.close()


def draw_graph(graph, dominating_set, filename):
    """Draws and saves a graph visualization with highlighted dominating set nodes."""
    G = nx.Graph(graph)  # Create graph from adjacency list

    plt.figure(figsize=(12, 12))  # Set figure size

    # Graph layout
    pos = nx.spring_layout(G, seed=42)  # Position nodes for visualization

    # Draw all nodes (default color)
    nx.draw(G, pos, with_labels=True, node_color="lightgray", edge_color="gray", node_size=800, font_size=12)

    # Highlight dominating set nodes in red
    nx.draw_networkx_nodes(G, pos, nodelist=dominating_set, node_color="red", node_size=900)

    # Save the figure
    plt.savefig(filename, format="png", bbox_inches="tight")
    plt.close()


def solve_dominating_set(graph):
    """
    Solves the Dominating Set problem using DOcplex with Branch and Bound.

    Parameters:
        graph (dict): Adjacency list representation of the graph.

    Returns:
        list: Minimum dominating set
    """

    mdl = Model("Dominating Set")

    mdl.parameters.mip.display = 0  # Suppresses verbose output

    # Create binary variables for each node
    nodes = list(graph.keys())
    x = {node: mdl.binary_var(name=f"x_{node}") for node in nodes}

    # Constraint: Each node is either in the set or has a neighbor in the set
    for node in nodes:
        mdl.add_constraint(x[node] + sum(x[neighbor] for neighbor in graph[node]) >= 1)

    # Objective: Minimize the number of nodes in the dominating set
    mdl.minimize(mdl.sum(x[node] for node in nodes))

    start_time = time.time()
    # Solve the model
    solution = mdl.solve(log_output=True)
    end_time = time.time()

    # Display only execution time
    print(f"Execution Time: {end_time - start_time:.4f} seconds")

    if solution:
        dominating_set = [node for node in nodes if x[node].solution_value > 0.5]
        return dominating_set
    else:
        return None  # No solution found


def is_valid_dominating_set(graph, dominating_set):
    """
    Verifies whether the given dominating_set is valid for the graph.

    Parameters:
    - graph (dict): Adjacency list representation of the graph.
    - dominating_set (list): Proposed dominating set.

    Returns:
    - bool: True if it's a valid dominating set, False otherwise.
    """
    covered_nodes = set(dominating_set)  # Nodes in the set should be covered

    # Add all neighbors of selected nodes (they are covered)
    for node in dominating_set:
        covered_nodes.update(graph[node])

    # Check if all nodes are covered
    return set(graph.keys()).issubset(covered_nodes)


def test_dominating_set_solver():
    with open("test_cases.json", "r") as json_file:
        test_graphs = json.load(json_file)
        test_graphs = test_graphs["test_graphs"]
        for i, test in enumerate(test_graphs):
            graph = {int(k): v for k, v in test["graph"].items()}
            expected = test["expected"]

            print(f"Test Case {i + 1}:")
            solution = solve_dominating_set(graph)  # Solve the problem

            print(f"  Graph: {graph}")
            print(f"  Expected: {expected}")
            print(f"  Found: {solution}")

            if is_valid_dominating_set(graph, solution):
                print("  ✅ Solution is valid!")
            else:
                print("  ❌ Solution is invalid!")

            # File names
            # expected_file = os.path.join("graph_images", f"test_{i + 1}_expected.png")
            # solution_file = os.path.join("graph_images", f"test_{i + 1}_solution.png")
            #
            # # Generate images
            # draw_graph(graph, expected, expected_file)
            # draw_graph(graph, solution, solution_file)

            # print(f"Saved: {expected_file} and {solution_file}")


# test_dominating_set_solver()


tests_dict_solution_ortools = {
    "test": [[1, 4], [2, 5]],
    "bremen_subgraph_20": [[4, 9, 13, 14, 15, 25, 27, 29, 30], [4, 9, 13, 15, 25, 26, 27, 29, 30],
                           [3, 8, 12, 13, 14, 24, 26, 28, 29], [3, 8, 12, 14, 24, 25, 26, 28, 29]],
    "bremen_subgraph_50": [[0, 7, 10, 13, 18, 23, 26, 27, 33, 34, 35, 40, 43, 45, 49, 59, 62],
                           [0, 7, 10, 13, 15, 23, 26, 27, 33, 34, 35, 40, 43, 45, 49, 59, 62]],
    "bremen_subgraph_100": [
        [1, 5, 13, 17, 21, 23, 25, 30, 31, 36, 38, 42, 47, 48, 50, 52, 54, 61, 67, 72, 78, 83, 84, 86, 87, 93, 98, 100,
         103],
        [1, 4, 8, 21, 22, 23, 26, 30, 31, 36, 38, 42, 47, 48, 50, 52, 54, 61, 67, 72, 78, 83, 84, 86, 87, 93, 98, 100,
         103]],
    "bremen_subgraph_150": [
        [1, 5, 6, 9, 14, 20, 25, 26, 29, 33, 35, 39, 43, 48, 52, 57, 63, 71, 73, 76, 81, 85, 87, 91, 98, 99, 102, 104,
         108, 113, 122, 128, 129, 133, 142, 145, 146, 147, 151, 154, 158, 163]],
    "bremen_subgraph_200": [
        [0, 2, 3, 6, 8, 11, 12, 21, 26, 32, 33, 35, 37, 38, 45, 49, 53, 54, 58, 60, 61, 73, 75, 78, 94, 97, 100, 101,
         102, 106, 107, 111, 117, 122, 125, 126, 127, 128, 131, 134, 140, 144, 152, 155, 158, 164, 170, 174, 178, 187,
         191, 195, 198, 202, 208, 210, 214]],
    "bremen_subgraph_250": [
        [0, 3, 5, 6, 10, 11, 16, 19, 25, 26, 28, 30, 38, 45, 47, 50, 53, 57, 65, 67, 70, 72, 76, 81, 86, 88, 92, 93, 98,
         99, 101, 107, 110, 112, 115, 119, 120, 132, 137, 140, 142, 145, 150, 163, 164, 168, 171, 176, 179, 181, 182,
         185, 187, 193, 196, 199, 204, 205, 211, 215, 220, 227, 232, 235, 236, 240, 241, 245, 250, 253, 258, 261, 264,
         267]],
    "bremen_subgraph_300": [
        [6, 7, 9, 10, 12, 13, 15, 19, 20, 21, 31, 34, 38, 40, 44, 47, 48, 52, 61, 68, 70, 73, 75, 82, 83, 84, 88, 89,
         96, 104, 110, 112, 113, 120, 123, 127, 129, 136, 138, 140, 143, 145, 148, 150, 155, 162, 166, 170, 173, 174,
         175, 178, 183, 186, 192, 193, 196, 199, 205, 207, 211, 213, 214, 216, 222, 227, 228, 235, 237, 239, 243, 245,
         252, 259, 262, 269, 275, 282, 288, 292, 295, 300, 305, 309],
        [5, 7, 9, 10, 12, 13, 15, 20, 24, 28, 31, 34, 38, 40, 42, 44, 45, 47, 48, 52, 58, 61, 63, 68, 70, 73, 82, 83,
         88, 89, 96, 100, 101, 104, 109, 110, 114, 117, 122, 127, 139, 143, 146, 148, 150, 151, 155, 162, 166, 169, 173,
         175, 178, 183, 186, 192, 193, 196, 199, 205, 207, 211, 213, 215, 217, 219, 221, 227, 228, 237, 240, 246, 247,
         259, 262, 275, 282, 290, 292, 295, 298, 300, 305, 308]],
    "test_isolated": [[0, 2]],
}

tests_dict_solution_manual = {
    "test": [[0, 4]],
    "bremen_subgraph_20": [[3, 4, 9, 11, 15, 23, 27, 29, 30], [2, 3, 8, 10, 14, 22, 26, 28, 29]],
    "bremen_subgraph_50": [[0, 1, 3, 5, 6, 7, 8, 10, 12, 14, 19, 21, 24, 35, 38, 41, 43, 48, 62]],
    "bremen_subgraph_100": [
        [0, 2, 4, 6, 8, 11, 14, 16, 17, 20, 23, 25, 27, 29, 31, 33, 35, 36, 37, 40, 42, 43, 45, 46, 47, 48, 50, 53, 54,
         57, 59, 65, 69, 72, 76, 77, 83, 86, 94, 97, 100, 103]],
    "bremen_subgraph_150": [
        [0, 1, 3, 4, 6, 7, 8, 11, 12, 13, 15, 16, 17, 19, 21, 23, 24, 29, 30, 31, 33, 34, 36, 38, 40, 43, 44, 47, 49,
         52, 54, 55, 56, 58, 59, 61, 63, 68, 71, 74, 76, 78, 82, 87, 90, 92, 95, 96, 98, 101, 103, 105, 107, 114, 115,
         130, 133, 136, 142, 148, 152, 157, 161]],
    "bremen_subgraph_200": [
        [0, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 23, 25, 28, 30, 32, 33, 35, 39, 41, 42, 43, 45,
         47, 49, 52, 55, 56, 58, 60, 61, 62, 64, 66, 71, 73, 76, 78, 79, 83, 84, 86, 88, 90, 92, 95, 97, 99, 103, 110,
         114, 119, 120, 123, 125, 127, 128, 130, 134, 135, 137, 139, 146, 150, 153, 155, 160, 163, 164, 173, 176, 179,
         192, 195, 199, 201, 212]],
    "bremen_subgraph_250": [
        [0, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 24, 25, 26, 29, 32, 34, 35, 38, 39, 41, 46, 47, 50, 55, 56,
         58, 60, 62, 65, 66, 68, 71, 74, 79, 80, 82, 83, 84, 86, 87, 91, 95, 96, 97, 99, 101, 102, 107, 109, 110, 112,
         114, 123, 125, 126, 129, 131, 132, 134, 135, 137, 139, 141, 144, 146, 147, 148, 150, 151, 155, 156, 158, 159,
         164, 166, 167, 169, 173, 175, 179, 181, 187, 189, 191, 193, 194, 200, 201, 203, 206, 209, 211, 214, 223, 226,
         242, 246, 247, 251, 254, 262, 265, 266]],
    "bremen_subgraph_300": [
        [0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 22, 25, 27, 30, 31, 33, 35, 36, 38, 39, 41, 45, 46, 49, 51,
         53, 54, 56, 57, 59, 61, 63, 66, 67, 69, 72, 75, 81, 84, 86, 87, 89, 91, 92, 93, 95, 97, 99, 101, 102, 104, 106,
         107, 109, 111, 113, 117, 121, 130, 132, 133, 136, 138, 139, 141, 142, 145, 147, 148, 149, 152, 153, 156, 161,
         163, 165, 166, 172, 174, 176, 180, 182, 188, 190, 194, 195, 197, 199, 203, 205, 206, 208, 209, 214, 216, 218,
         219, 222, 223, 227, 229, 235, 238, 243, 246, 247, 251, 253, 273, 275, 286, 288, 292, 302, 304, 307]],
    "test_isolated": [[0, 2]],
}

tests_dict_solution_docplex = {
    "test": [[2, 5]],
    "bremen_subgraph_20": [[4, 9, 13, 14, 15, 24, 27, 29, 30]],
    "bremen_subgraph_50": [[1, 8, 11, 14, 17, 19, 24, 27, 28, 34, 35, 36, 41, 44, 46, 50, 60]],
    "bremen_subgraph_100": [
        [2, 6, 7, 12, 14, 22, 24, 26, 31, 37, 39, 43, 48, 49, 53, 57, 58, 62, 64, 68, 73, 79, 84, 87, 88, 94, 99, 101,
         104]],
    "bremen_subgraph_150": [
        [2, 6, 10, 13, 15, 22, 23, 26, 28, 34, 36, 40, 44, 49, 53, 58, 64, 73, 74, 77, 82, 86, 88, 92, 99, 100, 103,
         107, 109, 110, 113, 123, 129, 130, 134, 143, 146, 147, 148, 152, 159, 162]],
    "bremen_subgraph_200": [
        [1, 2, 5, 7, 9, 11, 13, 18, 19, 28, 29, 33, 34, 35, 38, 39, 47, 50, 53, 55, 59, 62, 65, 66, 67, 75, 78, 94, 99,
         102, 104, 111, 113, 116, 118, 121, 126, 128, 129, 132, 136, 141, 146, 155, 162, 172, 176, 179, 182, 188, 192,
         196, 198, 203, 210, 211, 215]],
    "bremen_subgraph_250": [
        [1, 4, 6, 7, 11, 12, 17, 20, 26, 27, 29, 31, 46, 48, 51, 54, 58, 66, 68, 71, 73, 77, 82, 87, 89, 93, 94, 99,
         100, 102, 108, 111, 113, 116, 120, 121, 132, 133, 135, 138, 141, 143, 146, 151, 165, 168, 172, 176, 182, 183,
         184, 186, 188, 194, 197, 200, 205, 206, 212, 216, 218, 221, 228, 233, 236, 237, 241, 242, 246, 251, 254, 262,
         265, 268]],
    "bremen_subgraph_300": [
        [1, 6, 8, 10, 11, 13, 14, 16, 19, 21, 28, 32, 35, 39, 41, 43, 45, 48, 49, 53, 59, 62, 69, 71, 74, 83, 84, 89,
         90, 95, 101, 102, 105, 109, 111, 113, 114, 121, 123, 128, 140, 144, 147, 151, 152, 156, 159, 163, 171, 174,
         176, 179, 183, 187, 190, 195, 196, 198, 200, 206, 208, 212, 214, 216, 218, 220, 222, 228, 229, 238, 241, 247,
         248, 260, 263, 276, 283, 288, 291, 293, 299, 301, 306, 309]],
    "test_isolated": [[1, 3]],
}


def verify_solutions():
    with open("aux.json", "r") as json_file:
        test_graphs = json.load(json_file)
        test_graphs = test_graphs["test_graphs"]

        # verify or-tools solutions
        for test_case, solutions in tests_dict_solution_ortools.items():
            for sol in solutions:
                solution = [i+1 for i in sol]
                graph = {int(k): v for k, v in test_graphs[test_case].items()}
                if is_valid_dominating_set(graph, solution):
                    print(f"[OR-TOOLS]  ✅ Solution is valid for {test_case}!")
                else:
                    print(f"[OR-TOOLS]  ❌ Solution is invalid for {test_case}!")
                    print(solution)

        # verify manual solutions
        for test_case, solutions in tests_dict_solution_manual.items():
            for sol in solutions:
                solution = [i + 1 for i in sol]
                graph = {int(k): v for k, v in test_graphs[test_case].items()}
                if is_valid_dominating_set(graph, solution):
                    print(f"[MANUAL]  ✅ Solution is valid for {test_case}!")
                else:
                    print(f"[MANUAL]  ❌ Solution is invalid for {test_case}!")
                    print(solution)

        # verify docplex solutions
        for test_case, solutions in tests_dict_solution_docplex.items():
            for sol in solutions:
                solution = sol
                graph = {int(k): v for k, v in test_graphs[test_case].items()}
                if is_valid_dominating_set(graph, solution):
                    print(f"[DOCPLEX]  ✅ Solution is valid for {test_case}!")
                else:
                    print(f"[DOCPLEX]  ❌ Solution is invalid for {test_case}!")
                    print(solution)


# verify_solutions()


def run_multiple_times():
    with open("aux.json", "r") as json_file:
        test_graphs = json.load(json_file)
        test_graphs = test_graphs["test_graphs"]
        test_dict = {}
        for test_case, graph in test_graphs.items():
            graph = {int(k): v for k, v in graph.items()}
            print(test_case)
            test_dict[test_case] = {"solutions": [], "time": 0}
            sum_time = 0
            for i in range(18):
                start_time = time.time()
                solution = solve_dominating_set(graph)
                end_time = time.time()

                just_time = end_time - start_time
                sum_time += just_time

                test_dict[test_case]["solutions"] += solution,
            test_dict[test_case]["time"] = sum_time

        print("\n\n\n\n\n")
        for test_case, values in test_dict.items():
            print(test_case)
            for i in values["solutions"]:
                print(i)
            print(values["time"]/18)

run_multiple_times()

def generate_images():
    with open("test_cases.json", "r") as json_file:
        test_graphs = json.load(json_file)
        test_graphs = test_graphs["test_graphs"]
        for i, test in enumerate(test_graphs):
            graph = {int(k): v for k, v in test["graph"].items()}
            expected = test["expected"]
            solution_file = os.path.join("graph_images", "EXPECTED", f"test_{i+1}.png")
            draw_graph_v2(graph, expected, solution_file)

    with open("aux.json", "r") as json_file:
        test_graphs = json.load(json_file)
        test_graphs = test_graphs["test_graphs"]

        for test_case, solutions in tests_dict_solution_docplex.items():
            for i, sol in enumerate(solutions):
                solution = sol
                graph = {int(k): v for k, v in test_graphs[test_case].items()}
                solution_file = os.path.join("graph_images", "DOCPLEX", f"{test_case}_solution_{i}.png")
                draw_graph_v2(graph, solution, solution_file)

        for test_case, solutions in tests_dict_solution_ortools.items():
            for i, sol in enumerate(solutions):
                solution = [i + 1 for i in sol]
                graph = {int(k): v for k, v in test_graphs[test_case].items()}
                solution_file = os.path.join("graph_images", "ORTOOLS", f"{test_case}_solution_{i}.png")
                draw_graph_v2(graph, solution, solution_file)

        for test_case, solutions in tests_dict_solution_manual.items():
            for i, sol in enumerate(solutions):
                solution = [i + 1 for i in sol]
                graph = {int(k): v for k, v in test_graphs[test_case].items()}
                solution_file = os.path.join("graph_images", "MANUAL", f"{test_case}_solution_{i}.png")
                draw_graph_v2(graph, solution, solution_file)

# generate_images()
