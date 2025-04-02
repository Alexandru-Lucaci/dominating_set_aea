from docplex.cp.model import CpoModel

model = CpoModel()

N = 4

blocked_positions = [(0, 1), (0, 2)]

queens = [model.integer_var(0, N - 1, name=f"Q{i}") for i in range(N)]

# Constrangere pe coloane
model.add(model.all_diff(queens))

# Constarngere pe diagonale: principala, secundara
for i in range(N):
    for j in range(i + 1, N):
        model.add(queens[i] != queens[j] + (j - i))
        model.add(queens[i] != queens[j] - (j - i))

# Constarngere pozitii blocate
for row, col in blocked_positions:
    model.add(queens[row] != col)

solution = model.solve()

if solution:
    board = [["." for _ in range(N)] for _ in range(N)]
    for row, col in blocked_positions:
        board[row][col] = "X"

    for i in range(N):
        if (i, solution[queens[i]]) not in blocked_positions:
            board[i][solution[queens[i]]] = "Q"

    print("4×4 N-Queens Solution with Blocked Positions:")
    for row in board:
        print(" ".join(row))
else:
    print("No solution found.")