
import numpy as np
from alien import Alien
from typing import List, Tuple


def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):

    alien_shape = alien.get_shape()
    alien_centroid = alien.get_centroid()
    
    if alien_shape == 'Ball':
        r = alien.get_width()
        for w in walls:
            if point_segment_distance(alien_centroid, ((w[0], w[1]), (w[2], w[3]))) <= r:
                return True

    elif alien_shape in ['Horizontal', 'Vertical']:
        ht = alien.get_head_and_tail()
        for w in walls:
            if segment_distance(ht, ((w[0], w[1]), (w[2], w[3]))) <= alien.get_width():
                return True

    return False

def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    centroid = alien.get_centroid() 
    width = alien.get_width()

    if alien.get_shape() == 'Ball':
        conditiony = 0<centroid[1]+width<window[1] and 0<centroid[1]-width<window[1]
        conditionx = 0<centroid[0]+width<window[0] and 0<centroid[0]-width<window[0]
        return conditiony and conditionx
            
    elif alien.get_shape() == 'Horizontal':
        length = alien.get_length()
        return 0<centroid[0]+length/2 + width <window[0] and 0<centroid[0]-length/2 - width <window[0] and 0<centroid[1]+width <window[1] and 0<centroid[1]-width <window[1]
            
    elif alien.get_shape() == 'Vertical':
        length = alien.get_length()
        return 0<centroid[0]+width <window[0] and 0<centroid[0]-width <window[0] and 0<centroid[1]+length/2 + width <window[1] and 0<centroid[1]-length/2 - width <window[1]


def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (x1 == x and y1 == y) or (x2 == x and y2 == y) or (y1 == y == y2 and ((x1 <= x <= x2) or (x2 <= x <= x1))) or (x == x1 == x2 and ((y1 <= y <= y2) or (y2 <= y <= y1)) ):
            return True
        if (y1 < y and y2 >= y) or (y2 < y and y1 >= y):
            if (x1 + (y - y1) / (y2 - y1) * (x2 - x1)) < x:
                inside += 1
    return inside % 2 == 1




def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall.

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endy), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if the path touches a wall, False if not
    """
    if does_alien_touch_wall(alien, walls):
        return True

    if alien.is_circle():
        start_pos = alien.get_centroid()
        end_pos = waypoint

        if start_pos == end_pos:
            return False

        alien_width = alien.get_width()

        for wall_segment in walls:
            path = (start_pos, end_pos)
            obstacle = ((wall_segment[0], wall_segment[1]), (wall_segment[2], wall_segment[3]))

            if segment_distance(path, obstacle) <= alien_width:
                return True
    else:
        start_pos = alien.get_centroid()
        end_pos = waypoint

        if start_pos == end_pos:
            return False

        delta_coordinates = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])

        head_tail_positions = alien.get_head_and_tail()
        start_head = head_tail_positions[0]
        start_tail = head_tail_positions[1]
        end_head = (start_head[0] + delta_coordinates[0], start_head[1] + delta_coordinates[1])
        end_tail = (start_tail[0] + delta_coordinates[0], start_tail[1] + delta_coordinates[1])

        bounding_box = (start_head, start_tail, end_tail, end_head)

        if delta_coordinates[0] == 0 and start_head[0] == start_tail[0]:
            return handle_directional_movement(alien, walls, delta_coordinates)

        if delta_coordinates[1] == 0 and start_head[1] == start_tail[1]:
            return handle_directional_movement(alien, walls, delta_coordinates)

        for wall_segment in walls:
            wall_start = (wall_segment[0], wall_segment[1])
            wall_end = (wall_segment[2], wall_segment[3])

            if is_point_in_polygon(wall_start, bounding_box) or is_point_in_polygon(wall_end, bounding_box):
                return True

        alien_width = alien.get_width()

        for wall_segment in walls:
            wall_start = (wall_segment[0], wall_segment[1])
            wall_end = (wall_segment[2], wall_segment[3])

            if segment_distance((end_head, end_tail), (wall_start, wall_end)) <= alien_width:
                return True

        for wall_segment in walls:
            wall_start = (wall_segment[0], wall_segment[1])
            wall_end = (wall_segment[2], wall_segment[3])

            for i in range(len(bounding_box)):
                v1 = bounding_box[i]

                if i == len(bounding_box) - 1:
                    v2 = bounding_box[0]
                else:
                    v2 = bounding_box[i + 1]

                if do_segments_intersect((wall_start, wall_end), (v1, v2)):
                    return True

    return False

def handle_directional_movement(alien, walls, delta_coordinates):
    head_tail_positions = alien.get_head_and_tail()
    start_head = head_tail_positions[0]
    start_tail = head_tail_positions[1]
    end_head = (start_head[0] + delta_coordinates[0], start_head[1] + delta_coordinates[1])
    end_tail = (start_tail[0] + delta_coordinates[0], start_tail[1] + delta_coordinates[1])

    if delta_coordinates[0] == 0:
        index = 1
    else:
        index = 0

    difference_start_head_end_tail = abs(start_head[index] - end_tail[index])
    difference_start_tail_end_head = abs(start_tail[index] - end_head[index])

    line_segment = (start_head, end_tail) if difference_start_head_end_tail > difference_start_tail_end_head else (start_tail, end_head)

    for wall_segment in walls:
        wall_start = (wall_segment[0], wall_segment[1])
        wall_end = (wall_segment[2], wall_segment[3])

        if segment_distance(line_segment, (wall_start, wall_end)) <= alien.get_width():
            return True

    return False


def point_segment_distance(p, s):
  
    x, y = p
    x1, y1 = s[0]
    x2, y2 = s[1]

    segment_length = np.linalg.norm(np.array(s[1]) - np.array(s[0]))

    if segment_length == 0:
        return np.linalg.norm(np.array(p) - np.array(s[0]))

    t = np.dot(np.array([x - x1, y - y1]), np.array([x2 - x1, y2 - y1])) / (segment_length ** 2)

    if t < 0:
        return np.linalg.norm(np.array(p) - np.array(s[0]))
    elif t > 1:
        return np.linalg.norm(np.array(p) - np.array(s[1]))
    else:
        projection = np.array([x1, y1]) + t * np.array([x2 - x1, y2 - y1])
        return np.linalg.norm(np.array(p) - projection)

def do_segments_intersect(s1, s2):

    x1, y1 = s1[0]
    x2, y2 = s1[1]
    x3, y3 = s2[0]
    x4, y4 = s2[1]

    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    o1 = orientation((x1, y1), (x2, y2), (x3, y3))
    o2 = orientation((x1, y1), (x2, y2), (x4, y4))
    o3 = orientation((x3, y3), (x4, y4), (x1, y1))
    o4 = orientation((x3, y3), (x4, y4), (x2, y2))

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment((x1, y1), (x3, y3), (x2, y2)):
        return True
    if o2 == 0 and on_segment((x1, y1), (x4, y4), (x2, y2)):
        return True
    if o3 == 0 and on_segment((x3, y3), (x1, y1), (x4, y4)):
        return True
    if o4 == 0 and on_segment((x3, y3), (x2, y2), (x4, y4)):
        return True

    return False
def segment_distance(s1, s2):   

 
    x1, y1 = s1[0]
    x2, y2 = s1[1]
    x3, y3 = s2[0]
    x4, y4 = s2[1]


    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3


    determinant = dx1 * dy2 - dx2 * dy1

    if abs(determinant) < 1e-6:
        return np.min([point_segment_distance(s1[0], s2), point_segment_distance(s1[1], s2),
                       point_segment_distance(s2[0], s1), point_segment_distance(s2[1], s1)])

    t1 = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / determinant
    t2 = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / determinant

    if 0 <= t1 <= 1 and 0 <= t2 <= 1:
        return 0.0

    distances = [
        point_segment_distance(s1[0], s2),
        point_segment_distance(s1[1], s2),
        point_segment_distance(s2[0], s1),
        point_segment_distance(s2[1], s1)
    ]

    return np.min(distances)

if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")