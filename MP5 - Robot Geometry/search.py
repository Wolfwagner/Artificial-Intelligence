# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018


from collections import deque
import heapq
import maze

def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI

    
def astar(maze):
    start = maze.get_start()
    
    explored = set()
    frontier = [] 
    heapq.heappush(frontier, start)
    
    visited_states = {start:(None, 0)}   

    while frontier:
        current_state = heapq.heappop(frontier)
       

        if current_state.is_goal():
            return backtrack(visited_states, current_state)


        for neighbor in current_state.get_neighbors():
            g_value = neighbor.dist_from_start  
            
            if neighbor not in visited_states or g_value < visited_states[neighbor][1]:
                visited_states[neighbor] = (current_state,g_value)
                heapq.heappush(frontier, neighbor)

    return None

def backtrack(visited_states, current_state):
    path = []
    while current_state is not None:
        path.append(current_state)
        current_state, _ = visited_states[current_state]  
    return list(reversed(path))
