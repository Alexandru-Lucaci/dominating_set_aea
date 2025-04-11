import os

def parse_pace_input(filePath: str):
    """
    Parses a PACE-style Dominating Set input from stdin.
    Format:
      c (comment lines)
      p ds n m
      u v    (m lines of edges)
    Returns:
      n (int): number of vertices
      edges (list of tuples): list of undirected edges (1-based).
    """
    n = 0
    m = 0
    edges = []
    with open(filePath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            # logger.log(f"Line: {line}", level=logging.WARNING)
            parts = line.split()
            if parts[0] == 'p':
                # line looks like: p ds n m
                n = int(parts[2])
                m = int(parts[3])
            else:
                # edge line: u v
                u = int(parts[0])
                v = int(parts[1])
                edges.append((u, v))

    return n, edges


def get_test_files(directory):
    """
    Get all .gr test files in the specified directory.
    """
    return [filePath for filePath in os.listdir(directory) if filePath.endswith(".gr")]


def get_sol_files(directory):
    """
    Get all .sol solution files in the specified directory.
    """
    return [filePath for filePath in os.listdir(directory) if filePath.endswith(".sol")]
