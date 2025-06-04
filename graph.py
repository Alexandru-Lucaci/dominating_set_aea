class Graph:
    def __init__(self, n):
        """
        Create a graph with n vertices.
        PACE format uses 1-based indexing, but internally we use 0-based.
        So we'll have vertices 0 to n-1 internally.
        """
        self.n = n
        self.adjacency_list = [set() for _ in range(n)]
        self.jsonGraph = {}

    def add_edge(self, u, v):
        """
        Add undirected edge (u, v).
        u, v are 0-based indices (already converted from 1-based PACE format).
        """
        if u < self.n and v < self.n:
            self.adjacency_list[u].add(v)
            self.adjacency_list[v].add(u)
            
            if u not in self.jsonGraph:
                self.jsonGraph[u] = []
            self.jsonGraph[u].append(v)
                
            if v not in self.jsonGraph:
                self.jsonGraph[v] = []
            self.jsonGraph[v].append(u)

    def neighbors_of(self, v):
        """
        Return the set of neighbors of vertex v.
        """
        if v < self.n:
            return self.adjacency_list[v]
        return set()