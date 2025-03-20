from docplex.cp.model import CpoModel

model = CpoModel()

countries = ["Belgium", "Denmark", "France", "Germany", "Luxembourg", "Netherlands", "Switzerland"]

neighbors = {
    "Belgium": ["France", "Netherlands", "Luxembourg", "Germany"],
    "Denmark": ["Germany"],
    "France": ["Belgium", "Luxembourg", "Germany", "Switzerland"],
    "Germany": ["Belgium", "Denmark", "France", "Luxembourg", "Netherlands", "Switzerland"],
    "Luxembourg": ["Belgium", "France", "Germany"],
    "Netherlands": ["Belgium", "Germany"],
    "Switzerland": ["France", "Germany"]
}

colors = ["blue", "white", "yellow", "green"]

color_vars = {c: model.integer_var(min=0, max=3, name=c) for c in countries}

for country, neighbors_list in neighbors.items():
    for neighbor in neighbors_list:
        if country < neighbor and not (
                (country == "Germany" and neighbor == "Denmark") or
                (country == "Denmark" and neighbor == "Germany")):
            model.add(color_vars[country] != color_vars[neighbor])

model.add(color_vars["Germany"] == color_vars["Denmark"])

solution = model.solve()

if solution:
    for country in countries:
        print(f"{country}: Color {colors[solution[color_vars[country]]]}")
else:
    print("No solution found.")
