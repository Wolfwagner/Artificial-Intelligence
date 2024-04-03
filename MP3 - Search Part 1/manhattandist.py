def manhattan_distance(state, goal):
    """
    Calculate the Manhattan distance between two 8-puzzle states.

    Parameters:
    - state: The current state of the puzzle as a 2D list.
    - goal: The goal state of the puzzle as a 2D list.

    Returns:
    - The Manhattan distance as an integer.
    """
    size = len(state)
    distance = 0

    for i in range(size):
        for j in range(size):
            tile = state[i][j]
            if tile != 0:  # Exclude the empty tile (0)
                goal_x, goal_y = find_goal_position(tile, goal)
                distance += abs(i - goal_x) + abs(j - goal_y)

    return distance

def find_goal_position(tile, goal):
    """
    Find the goal position (row, column) for a given tile value in the goal state.

    Parameters:
    - tile: The value of the tile.
    - goal: The goal state of the puzzle as a 2D list.

    Returns:
    - The goal position as a tuple (row, column).
    """
    size = len(goal)
    for i in range(size):
        for j in range(size):
            if goal[i][j] == tile:
                return i, j

def get_neighbors(state):
    """
    Generate neighboring states for an 8-puzzle state by moving the empty tile.

    Parameters:
    - state: The current state of the puzzle as a 2D list.

    Returns:
    - A list of neighboring states (2D lists).
    """
    neighbors = []
    size = len(state)

    # Find the position of the empty tile (0)
    for i in range(size):
        for j in range(size):
            if state[i][j] == 0:
                empty_row, empty_col = i, j

    # Possible moves (up, down, left, right)
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    for dr, dc in moves:
        new_row, new_col = empty_row + dr, empty_col + dc

        if 0 <= new_row < size and 0 <= new_col < size:
            # Create a copy of the current state
            new_state = [row[:] for row in state]

            # Swap the empty tile and the neighboring tile
            new_state[empty_row][empty_col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[empty_row][empty_col]

            neighbors.append(new_state)

    return neighbors

# Example usage:
initial_state = [
    [1, 2, 3],
    [0, 4, 5],
    [6, 7, 8]
]

goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

# Get neighboring states
neighbor_states = get_neighbors(initial_state)
print(neighbor_states)

for neighbor in neighbor_states:
    distance = manhattan_distance(neighbor,initial_state)
    print(distance)

