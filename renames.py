import os
import shutil

def copy_and_rename_data_files(base_path="results", output_path="resultsRenamed"):
    """
    Copy data.csv files to a new folder with renamed filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    for root, dirs, files in os.walk(base_path):
        if "data.csv" in files:
            # Extract the relevant parts from the path
            path_parts = root.replace("\\", "/").split("/")

            # Find the subgraph and algorithm parts
            subgraph_part = None
            algorithm_part = None

            for part in path_parts:
                if "subgraph" in part:
                    subgraph_part = part
                elif part in ["tabu_search", "orTools", "ourSolution","fast_tabu_search"]:  # add other algorithms as needed
                    algorithm_part = part

            if subgraph_part and algorithm_part:
                # Create new filename
                new_filename = f"{subgraph_part}__{algorithm_part}__data.csv"

                # Full paths
                source_path = os.path.join(root, "data.csv")
                destination_path = os.path.join(output_path, new_filename)

                # Copy the file with new name
                try:
                    shutil.copy2(source_path, destination_path)
                    print(f"Copied: {source_path} -> {destination_path}")
                except Exception as e:
                    print(f"Error copying {source_path}: {e}")

# Run the function
copy_and_rename_data_files()