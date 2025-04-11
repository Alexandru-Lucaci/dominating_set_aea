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
        # Create Boolean (0-1) decision variables x[v]
        self.x_vars = [self.model.NewBoolVar(f'x_{v}') for v in range(self.n)]

        # Add domination constraints
        for i in range(self.n):
            # x[i] + sum(x[j] for j in neighbors_of(i)) >= 1
            neighbor_vars = [self.x_vars[j] for j in self.graph.neighbors_of(i)]
            self.model.Add(self.x_vars[i] + sum(neighbor_vars) >= 1)

        # Objective: minimize sum(x[v])
        self.model.Minimize(sum(self.x_vars))

    def solve(self, time_limit=None):
        """
        Solve the model, optionally with a time limit (in seconds).
        Returns a list of chosen vertices (0-based) forming a minimum DS if found.
        """
        solver = cp_model.CpSolver()

        # Optionally set a time limit if desired
        if time_limit is not None:
            solver.parameters.max_time_in_seconds = time_limit

        # Solve
        status = solver.Solve(self.model)

        # If a feasible or optimal solution was found, retrieve it
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            chosen = [v for v in range(self.n) if solver.Value(self.x_vars[v]) == 1]
            return chosen
        else:
            # No feasible solution found (should be rare for DS)
            return []
