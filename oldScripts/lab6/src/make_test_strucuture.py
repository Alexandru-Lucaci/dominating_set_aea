import json
import os


def read_graph_from_file(filename):
    """Reads a .gr file and returns an adjacency list (graph dictionary)."""
    with open(filename, "r") as file:
        num_nodes, num_edges = map(int, file.readline().split())  # Read first line
        graph = {i: [] for i in range(1, num_nodes + 1)}  # Initialize graph

        for line in file:  # Read edges
            u, v = map(int, line.split())
            graph[u].append(v)
            graph[v].append(u)  # Undirected graph

    return graph


def read_solution_from_file(filename):
    """Reads a .sol file and returns the expected dominating set."""
    with open(filename, "r") as file:
        return [int(line.strip()) for line in file]  # Read each line as an integer


def generate_test_json(data_folder, output_file="test_cases.json"):
    """Processes all .gr and .sol files in the same folder and saves them as JSON."""
    test_graphs = []

    # Loop through all files in the folder
    for file in os.listdir(data_folder):
        if file.endswith(".gr"):  # Find all .gr files
            base_name = os.path.splitext(file)[0]  # Get filename without extension
            sol_file = os.path.join(data_folder, f"{base_name}.sol")  # Expected .sol file

            if os.path.exists(sol_file):  # Ensure solution file exists
                graph = read_graph_from_file(os.path.join(data_folder, file))
                expected = read_solution_from_file(sol_file)

                test_graphs.append({"graph": graph, "expected": expected})

    # Save test graphs to a JSON file
    with open(output_file, "w") as json_file:
        json.dump({"test_graphs": test_graphs}, json_file, indent=None, separators=(",", ":"))

    print(f"JSON file '{output_file}' created successfully!")


# Example usage
generate_test_json("graphs_test_data/")  # Replace "graphs_data/" with your actual folder path
