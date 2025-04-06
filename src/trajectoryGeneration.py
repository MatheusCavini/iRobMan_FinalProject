import numpy as np
import pybullet as p

#[MC:2025-02-15] Some functions to test how well the Jacobian Controller performs in interpolated trajectories
###########################################################
def interpolateLinearTrajectory(initial_position, initial_orientation, end_position, end_orientation, n_points):
    positions = []
    orientations = []
    for i in range(n_points):
        position = ((np.array(end_position) - np.array(initial_position))/n_points * i) + np.array(initial_position)
        orientation = np.array(p.getQuaternionSlerp(initial_orientation, end_orientation, i/n_points))

        positions.append(position)
        orientations.append(orientation) 

    return positions, orientations


def interpolateTrajectoryWaypoins(position_waypoints, orientation_waypoints, n_points):
    positions = []
    orientations = []
    for i in range(len(position_waypoints)-1):
        pos, ori = interpolateLinearTrajectory(position_waypoints[i], orientation_waypoints[i], position_waypoints[i+1], orientation_waypoints[i+1], int(n_points/(len(position_waypoints)-1)))
        positions += pos
        orientations += ori

    return positions, orientations




def moveAlongTrajectory(robot, positions, orientations, step_start, current_step):
    steps_duration = len(positions)
    if current_step > step_start and current_step < step_start+steps_duration:
        robot.move_to_pose(positions[current_step-step_start], orientations[current_step-step_start])
    
    if current_step >= step_start + steps_duration:
        return True
    else: 
        return False


###########################################################


#[MC:2025-03-27] Implementation of the RRT* algorithm for 3D path planning
###########################################################

import numpy as np
import random
from scipy.spatial import KDTree

class Node:
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent
        self.cost = 0

class RRTStar3D:
    def __init__(self, start, goal, world_bounds, max_iter=1000, step_size=0.01, search_radius=0.5, goal_sample_rate=0.1):
        self.start = Node(start)
        self.goal = Node(goal)
        self.world_bounds = world_bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.search_radius = search_radius
        self.goal_sample_rate = goal_sample_rate
        self.tree = [self.start]

    def sample_point(self):
        if random.random() < self.goal_sample_rate:
            return self.goal.position
        return np.array([random.uniform(b[0], b[1]) for b in self.world_bounds])

    def nearest_node(self, sampled_point):
        tree_points = np.array([node.position for node in self.tree])
        kd_tree = KDTree(tree_points)
        _, idx = kd_tree.query(sampled_point)
        return self.tree[idx]

    def steer(self, nearest, sampled):
        direction = sampled - nearest.position
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None
        direction = direction / norm
        new_position = nearest.position + self.step_size * direction
        return Node(new_position, parent=nearest)

    def is_collision_free(self, start, end):
        ray_test = p.rayTest(start, end)
        return ray_test[0][0] == -1

    def update_descendant_costs(self, node):
        for child in self.tree:
            if child.parent == node:
                child.cost = node.cost + np.linalg.norm(child.position - node.position)
                self.update_descendant_costs(child)

    def rewire(self, new_node):
        for node in self.tree:
            distance = np.linalg.norm(node.position - new_node.position)
            if node == new_node or distance > self.search_radius:
                continue
            new_cost = new_node.cost + distance
            if new_cost < node.cost and self.is_collision_free(new_node.position, node.position):
                node.parent = new_node
                node.cost = new_cost
                self.update_descendant_costs(node)

    def try_connect_goal(self, new_node):
        distance = np.linalg.norm(new_node.position - self.goal.position)
        if distance < self.search_radius and self.is_collision_free(new_node.position, self.goal.position):
            new_cost = new_node.cost + distance
            if self.goal.parent is None or new_cost < self.goal.cost:
                self.goal.parent = new_node
                self.goal.cost = new_cost
                return True
        return False

    def run(self, num_points):
        goal_connected = False

        for _ in range(self.max_iter):
            sampled_point = self.sample_point()
            nearest = self.nearest_node(sampled_point)
            new_node = self.steer(nearest, sampled_point)
            if new_node is None:
                continue

            if self.is_collision_free(nearest.position, new_node.position):
                new_node.cost = nearest.cost + np.linalg.norm(new_node.position - nearest.position)
                self.tree.append(new_node)
                self.rewire(new_node)

                if self.try_connect_goal(new_node):
                    goal_connected = True

        if goal_connected:
            self.tree.append(self.goal)
            path = self.extract_path()
            path = self.shortcut_path(path)
            return self.interpolate_path(path, num_points=num_points)
        else:
            return None  # No path found

    def extract_path(self):
        path = []
        node = self.goal
        while node is not None:
            path.append(node.position)
            node = node.parent
        return path[::-1]

    def shortcut_path(self, path):
        if len(path) < 3:
            return path
        new_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.is_collision_free(path[i], path[j]):
                    break
                j -= 1
            new_path.append(path[j])
            i = j
        return new_path

    def interpolate_path(self, path, num_points=100):
        if len(path) < 2:
            return path

        total_length = sum(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i])) for i in range(len(path) - 1))
        distances = [0]
        for i in range(1, len(path)):
            distances.append(distances[-1] + np.linalg.norm(np.array(path[i]) - np.array(path[i - 1])))

        interpolated_path = []
        target_distances = np.linspace(0, total_length, num_points)

        path_idx = 0
        for target_d in target_distances:
            while path_idx < len(distances) - 1 and distances[path_idx + 1] < target_d:
                path_idx += 1
            p1, p2 = np.array(path[path_idx]), np.array(path[path_idx + 1])
            segment_length = distances[path_idx + 1] - distances[path_idx]
            if segment_length == 0:
                interpolated_path.append(p1.tolist())
            else:
                alpha = (target_d - distances[path_idx]) / segment_length
                new_point = (p1 + alpha * (p2 - p1)).tolist()
                interpolated_path.append(new_point)

        return interpolated_path
    