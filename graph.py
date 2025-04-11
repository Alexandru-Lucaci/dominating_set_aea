class Graph:
    def __init__(self, n):
        """
        Create a graph with n vertices (0-based).
        Adjacency is stored in a list of sets for quick neighbor lookup.
        """
        self.n = n
        self.adjacency_list = [set() for _ in range(n)]
        self.jsonGraph = {}

    def add_edge(self, u, v):
        """
        Add undirected edge (u, v).
        u, v are assumed to be 0-based indices.
        """
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)
        u = u + 1
        v = v + 1
        if self.jsonGraph.get(u) is None:
            self.jsonGraph[u] = []
            self.jsonGraph[u].append(v)
        else:
            self.jsonGraph[u].append(v)
        if self.jsonGraph.get(v) is None:
            self.jsonGraph[v] = []
            self.jsonGraph[v].append(u)
        else:
            self.jsonGraph[v].append(u)

    def neighbors_of(self, v):
        """
        Return the set of neighbors of vertex v.
        """
        return self.adjacency_list[v]
