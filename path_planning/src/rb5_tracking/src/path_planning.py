import numpy as np
import heapq
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def generate_map_with_obstacle(width, height, resolution, obstacle_center, obstacle_size):
    """
    Generate a map with a rectangular obstacle.

    :param width: Width of the map.
    :param height: Height of the map.
    :param resolution: Resolution of the map (grid size).
    :param obstacle_center: Center coordinates of the obstacle.
    :param obstacle_size: Size (width, height) of the obstacle.
    :return: A numpy array representing the map with the obstacle.
    """
    # Calculate map size and obstacle dimensions based on resolution
    map_size = (height // resolution, width // resolution)
    obstacle_size_scaled = (obstacle_size[0] // resolution, obstacle_size[1] // resolution)
    obstacle_center_scaled = (obstacle_center[0] // resolution, obstacle_center[1] // resolution)

    # Initialize map and define obstacle boundaries
    map = np.zeros(map_size)
    top_left = (obstacle_center_scaled[0] - obstacle_size_scaled[0] // 2, 
                obstacle_center_scaled[1] - obstacle_size_scaled[1] // 2)
    bottom_right = (obstacle_center_scaled[0] + obstacle_size_scaled[0] // 2, 
                    obstacle_center_scaled[1] + obstacle_size_scaled[1] // 2)

    # Place the obstacle on the map
    map[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 1
    return map

def obstacle_dilation_circular(map, dilation_radius):
    """
    Dilate the obstacles in the map by a specified radius to account for robot size.

    :param map: The original map with obstacles.
    :param dilation_radius: Radius for dilation (to account for robot size).
    :return: A new map with dilated obstacles.
    """
    dilated_map = map.copy()
    # Dilate each obstacle point in the map
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i, j] == 1:
                for di in range(-dilation_radius, dilation_radius + 1):
                    for dj in range(-dilation_radius, dilation_radius + 1):
                        if di*di + dj*dj <= dilation_radius*dilation_radius:
                            if 0 <= i + di < map.shape[0] and 0 <= j + dj < map.shape[1]:
                                dilated_map[i + di, j + dj] = 1
    return dilated_map

def heuristic(a, b):
    """
    Heuristic function for A* search, calculating Euclidean distance.

    :param a: Tuple (x, y) as the current node.
    :param b: Tuple (x, y) as the goal node.
    :return: Euclidean distance between a and b.
    """
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def a_star_search(map, start, goal):
    """
    Perform A* search algorithm to find a path from start to goal.

    :param map: The map with obstacles.
    :param start: Starting point (x, y).
    :param goal: Goal point (x, y).
    :return: A list of tuples representing the path from start to goal.
    """
    # Define neighbor positions (8 directions)
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            # Reconstruct the path from goal to start
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        close_set.add(current)
        # Explore neighbors
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            # Check if neighbor is within map boundaries and not in an obstacle
            if 0 <= neighbor[0] < map.shape[0] and 0 <= neighbor[1] < map.shape[1] and map[neighbor[0]][neighbor[1]] != 1:
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                # Update path to neighbor if shorter
                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
                
    return []

def calculate_potential_field(map, goal, attractive_scale=1.0, repulsive_scale=20.0, repulsive_threshold=10):
    """
    Calculate the potential field for path planning.

    :param map: The map with obstacles.
    :param goal: The goal point (x, y).
    :param attractive_scale: Scaling factor for attractive potential.
    :param repulsive_scale: Scaling factor for repulsive potential.
    :param repulsive_threshold: Threshold distance for repulsive potential influence.
    :return: A numpy array representing the potential field across the map.
    """
    potential_field = np.zeros_like(map, dtype=np.float32)
    
    # Calculate potential field value for each point in the map
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            # Add attractive potential from the goal
            potential_field[i, j] += attractive_scale * heuristic((i, j), goal)
            # Add repulsive potential from the obstacles
            if map[i, j] == 1:
                for di in range(-repulsive_threshold, repulsive_threshold + 1):
                    for dj in range(-repulsive_threshold, repulsive_threshold + 1):
                        if 0 <= i + di < map.shape[0] and 0 <= j + dj < map.shape[1]:
                            distance = heuristic((i, j), (i + di, j + dj))
                            if distance <= repulsive_threshold and distance > 0:
                                potential_field[i + di, j + dj] += repulsive_scale * (1 / (distance) - 1 / repulsive_threshold)

    return potential_field

def apf_path_planning(map, start, goal, attractive_scale=1.0, repulsive_scale=20.0, repulsive_threshold=10, max_steps=1000):
    """
    Perform Artificial Potential Field (APF) path planning.

    :param map: The map with obstacles.
    :param start: Starting point (x, y).
    :param goal: Goal point (x, y).
    :param attractive_scale: Scaling factor for attractive potential.
    :param repulsive_scale: Scaling factor for repulsive potential.
    :param repulsive_threshold: Threshold distance for repulsive potential influence.
    :param max_steps: Maximum steps for the path planning algorithm.
    :return: A list of tuples representing the path and the potential field.
    """
    potential_field = calculate_potential_field(map, goal, attractive_scale, repulsive_scale, repulsive_threshold)
    path = []
    current = start

    for _ in range(max_steps):
        min_potential = float('inf')
        next_step = None

        # Check all adjacent cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = current[0] + dx, current[1] + dy
                # Choose the cell with the lowest potential that is not an obstacle
                if 0 <= nx < map.shape[0] and 0 <= ny < map.shape[1]:
                    if potential_field[nx, ny] < min_potential and map[nx, ny] == 0:
                        min_potential = potential_field[nx, ny]
                        next_step = (nx, ny)
        
        if next_step is None:
            print('No APF path found!')
            break
        
        if next_step == goal:
            path.append(next_step)
            break
        
        current = next_step
        path.append(current)

    return path, potential_field

def smooth_path_and_calculate_angles(path, resolution=0.2):
    """
    Smooth the path using cubic spline interpolation and calculate the angle at each point.

    :param path: The original path as a list of tuples (x, y).
    :param resolution: Resolution for generating the smooth path.
    :return: A tuple of two lists: the smooth path and the angles at each point.
    """
    if not path:
        return [], []

    # Extract x and y coordinates
    x, y = zip(*path)

    # Generate interpolation parameters
    t = np.linspace(0, 1, len(path))

    # Create cubic spline interpolation
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)

    # Generate interpolation points at a finer interval
    t_fine = np.arange(0, 1, resolution)
    x_fine = cs_x(t_fine)
    y_fine = cs_y(t_fine)

    # Ensure the interpolation path includes the last point
    if (x_fine[-1], y_fine[-1]) != path[-1]:
        x_fine = np.append(x_fine, path[-1][0])
        y_fine = np.append(y_fine, path[-1][1])

    # Calculate tangent angles
    angles = np.degrees(np.arctan2(cs_y.derivative()(t_fine), cs_x.derivative()(t_fine)))

    # Return the smooth path and corresponding angles
    smooth_path = list(zip(x_fine, y_fine))
    return smooth_path, angles


# Main
if __name__ == "__main__":
    # Define map parameters
    width, height, resolution = 200, 200, 5
    obstacle_center = (100, 100)
    obstacle_size = (30, 30)

    # Generate map with an obstacle
    map = generate_map_with_obstacle(width, height, resolution, obstacle_center, obstacle_size)

    # Define robot parameters and start/goal points
    robot_radius = 30 // resolution
    start = (150 // resolution, 50 // resolution)
    goal = (55 // resolution, 155 // resolution)

    # Dilate obstacles considering robot size
    dilated_map = obstacle_dilation_circular(map, robot_radius)
    print(dilated_map)

    # Perform A* pathfinding
    a_star_path = a_star_search(dilated_map, start, goal)

    # Perform APF path planning
    apf_path, potential_field = apf_path_planning(map, start, goal, attractive_scale=1.0, repulsive_scale=20.0, repulsive_threshold=robot_radius, max_steps=1000)

    # Smooth the paths and calculate angles
    smooth_a_star_path, a_star_angles = smooth_path_and_calculate_angles(a_star_path)
    smooth_apf_path, apf_angles = smooth_path_and_calculate_angles(apf_path)

    # Calculate path lengths
    a_star_path_length = sum(heuristic(smooth_a_star_path[i], smooth_a_star_path[i + 1]) for i in range(len(smooth_a_star_path) - 1))
    apf_path_length = sum(heuristic(smooth_apf_path[i], smooth_apf_path[i + 1]) for i in range(len(smooth_apf_path) - 1))

    # Project to real-world coordinates
    a_star_path_real = [(x * resolution, y * resolution) for x, y in smooth_a_star_path]
    apf_path_real = [(x * resolution, y * resolution) for x, y in smooth_apf_path]

    a_star_path_output = [(x[0], x[1], y) for x , y in zip(a_star_path_real, [90]+list(a_star_angles))]
    apf_path_output = [(x[0], x[1], y) for x , y in zip(apf_path_real, [90]+list(apf_angles))]

    with open("/root/rb5_ws/src/rb5_ros/rb5_tracking/src/a_star_path.pkl", "wb") as file:
        pickle.dump(a_star_path_output, file)

    with open("/root/rb5_ws/src/rb5_ros/rb5_tracking/src/apf_path.pkl", "wb") as file:
        pickle.dump(apf_path_output, file)

    # Print results
    print("A* Path: {}".format(["({:.1f}, {:.1f})".format(x, y) for x, y in a_star_path_real]))
    print("A* Path Angles: {}".format(["({:.1f})".format(x) for x in a_star_angles]))
    print("APF Path: {}".format(["({:.1f}, {:.1f})".format(x, y) for x, y in apf_path_real]))
    print("APF Path Angles: {}".format(["({:.1f})".format(x) for x in apf_angles]))
    print("length:", len(apf_path_real), len(apf_angles))
    print("A* Path Length: {:.1f}".format(a_star_path_length * resolution))
    print("APF Path Length: {:.1f}".format(apf_path_length * resolution))

    # Visualization of the paths
    plt.figure(figsize=(15, 7))

    # Visualize A* Path
    plt.subplot(1, 2, 1)
    plt.title("A* Path")
    plt.imshow(dilated_map, cmap='gray')
    plt.plot([x for x, y in smooth_a_star_path], [y for x, y in smooth_a_star_path], 'ro-')
    plt.plot([x for x, y in a_star_path], [y for x, y in a_star_path], 'b--')
    plt.ylim(min(plt.ylim()), max(plt.ylim()))

    # Visualize APF Path
    plt.subplot(1, 2, 2)
    plt.title("APF Path")
    plt.imshow(potential_field, cmap='hot')
    plt.plot([x for x, y in smooth_apf_path], [y for x, y in smooth_apf_path], 'ro-')
    plt.plot([x for x, y in apf_path], [y for x, y in apf_path], 'b--')
    plt.ylim(min(plt.ylim()), max(plt.ylim()))

    plt.tight_layout()
    plt.show()
    plt.savefig("/root/rb5_ws/src/rb5_ros/rb5_tracking/src/path_planning.jpg")