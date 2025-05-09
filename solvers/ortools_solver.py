import subprocess

from graph import Graph

try:
    from ortools.sat.python import cp_model
except ImportError:
    subprocess.run(["pip", "install", "ortools"])
    from ortools.sat.python import cp_model



class ORToolsDominatingSetSolver:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.n = graph.n
        self.model = cp_model.CpModel()
        self.x_vars = []

    def build_model(self):
        """
        Build the CP-SAT model for the Dominating Set problem:
          Minimize sum(x[v]) subject to: for each vertex i,
          x[i] + sum(x[j] for j in neighbors(i)) >= 1
        """
        self.x_vars = [self.model.NewBoolVar(f'x_{v}') for v in range(self.n)]

        for i in range(self.n):
            neighbor_vars = [self.x_vars[j] for j in self.graph.neighbors_of(i)]
            self.model.Add(self.x_vars[i] + sum(neighbor_vars) >= 1)

        self.model.Minimize(sum(self.x_vars))

    def solve(self, time_limit=None):
        """
        Solve the model, optionally with a time limit (in seconds).
        Returns a list of chosen vertices (0-based) forming a minimum DS if found.
        """
        solver = cp_model.CpSolver()

        if time_limit is not None:
            solver.parameters.max_time_in_seconds = time_limit

        status = solver.Solve(self.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            chosen = [v for v in range(self.n) if solver.Value(self.x_vars[v]) == 1]
            return chosen
        else:
            return []
