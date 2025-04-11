import os
import subprocess

def draw_graph(testFilePath: str, dominating_set: list = None, title: str = "Graph", filePath: str = "graph.png"):
    """
    Draw the graph with the dominating set highlighted.
    
    Args:
        testFilePath: Path to the test file containing the graph
        dominating_set: List of vertices in the dominating set (0-based)
        title: Title for the graph
        filePath: Output file path for the graph image
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        subprocess.run(["pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt
    try:
        import networkx as nx
    except ImportError:
        subprocess.run(["pip", "install", "networkx"])
        import networkx as nx
        
    from ..utils.parser import parse_pace_input
    
    # Convert to 1-based for display
    dominating_set = [int(x + 1) for x in dominating_set]
    
    n, edges = parse_pace_input(testFilePath)
    G = nx.Graph()
    for u in range(n):
        G.add_node(u)
    for u, v in edges:
        G.add_edge(u, v)
    # remove vertex 0
    G.remove_node(0)
    
    colorList = ["red" if v in dominating_set else "blue" for v in G.nodes]
    print(f"Color List: {colorList}")
    print(f"Nodes: {G.nodes}")
    print(f"Dominating Set: {dominating_set}")

    # add title to the graph
    plt.title(f"Test Case: {title}")
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=0.05, iterations=1500)
    nx.draw(G, pos=pos, with_labels=True, node_color=colorList)

    # Create output directories if they don't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists(f"results/{title}"):
        os.makedirs(f"results/{title}")

    plt.savefig(f"{filePath}")
    plt.clf()
