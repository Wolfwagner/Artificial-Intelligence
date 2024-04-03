def get_neighbors(state):
    nbr_states = []  # Initialize an empty list to store neighboring states
    x, y = None, None# Get the current position (row, column) of the empty tile (0)
    for i in range(len(state[0])):
        for j in range(len(state[1])):
            if state[i][j] == 0:
                x,y= i,j
    moves = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # Define possible movements: down, left, up, and right

    for dx, dy in moves:
        new_x, new_y = x + dx, y + dy  # Calculate the new position by adding the movement (dx, dy)
        if 0 <= new_x < 3 and 0 <= new_y < 3:  # Check if the new position is within the 3x3 grid boundaries
            new_state = [row[:] for row in state]  # Create a copy of the current state
            new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]  # Swap the empty tile with the adjacent tile
            nbr_states.append(new_state) # Create a new EightPuzzleState object with the updated state and other parameters and add it to the list of neighbors

    return nbr_states

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])



def compute_heuristic(state):
    heuristic = 0
    size = len(state)
    
    for i in range(size):
            for j in range(size):
                tile = state[i][j]

                if tile != 0:  # Ignore the empty tile (0)
                    goal_x, goal_y = (tile - 1) // size, (tile - 1) % size
                    current_position = (i, j)
                    manhattan_distance = manhattan(current_position, (goal_x, goal_y))
                    heuristic += manhattan_distance

    return heuristic


if __name__ == "__main__":
    
    state= [
    [1, 2, 3],
    [0, 4, 5],
    [6, 7, 8]
]
    neighbour_states= get_neighbors(state)
    print(neighbour_states)
    
    for neighbor in neighbour_states:
        distance= compute_heuristic(neighbor)
        print(distance)
