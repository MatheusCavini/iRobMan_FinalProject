import numpy as np
import open3d as o3d
from typing import Any, Sequence, Optional

import open3d as o3d
import numpy as np

from typing import Tuple, Sequence, Optional

def create_grasp_mesh(
    center_point: np.ndarray,
    width: float = 0.05,
    height: float = 0.1,
    depth: float = 0.03,
    gripper_distance: float = 0.15,
    gripper_height: float = 0.1,
    color_left: list = [1, 0, 0],  # Red
    color_right: list = [0, 1, 0],  # Green,
    scale: float = 0.5,
    rotation_matrix: Optional[np.ndarray] = None
) -> Sequence[o3d.geometry.TriangleMesh]:
    """
    Creates a mesh representation of a robotic gripper.

    Args:
        center_point: Central position of the gripper in 3D space
        width: Width of each gripper finger
        height: Height of each gripper finger
        depth: Depth of each gripper finger
        gripper_distance: Distance between gripper fingers
        gripper_height: Height of the gripper base
        color_left: RGB color values for left finger [0-1]
        color_right: RGB color values for right finger [0-1]
        scale: Scaling factor for the gripper dimensions
        rotation_matrix: Optional 3x3 rotation matrix for gripper orientation

    Returns:
        list: List of mesh geometries representing the gripper components
    """
    grasp_geometries = []

    # Apply scaling to dimensions
    width *= scale
    height *= scale
    depth *= scale
    gripper_distance *= scale
    gripper_height *= scale

    # Create left finger
    left_finger = o3d.geometry.TriangleMesh.create_box(
        width=width/2,
        height=height,
        depth=depth
    )
    left_finger.paint_uniform_color(color_left)
    left_finger.translate((-gripper_distance-width/2, 0, 0) + center_point)
    if rotation_matrix is not None:
        left_finger.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(left_finger)

    # Create right finger
    right_finger = o3d.geometry.TriangleMesh.create_box(
        width=width/2,
        height=height,
        depth=depth
    )
    right_finger.paint_uniform_color(color_right)
    right_finger.translate((gripper_distance, 0, 0) + center_point)
    if rotation_matrix is not None:
        right_finger.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(right_finger)

    coupler = o3d.geometry.TriangleMesh.create_box(
        width=2*gripper_distance + width,
        height=width/2,
        depth=depth
    )
    coupler.paint_uniform_color([0, 0, 1])
    coupler.translate(
        (-gripper_distance-width/2, gripper_height, 0) + center_point)
    if rotation_matrix is not None:
        coupler.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(coupler)

    stick = o3d.geometry.TriangleMesh.create_box(
        width=width/2,
        height=height*1.5,
        depth=depth
    )
    stick.paint_uniform_color([0, 0, 1])
    stick.translate((-width/4, gripper_height, 0) + center_point)
    if rotation_matrix is not None:
        stick.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(stick)

    return grasp_geometries

def exp_map_so3(axis: np.ndarray, angle: float) -> np.ndarray:
    # ---COMPLETE-THE-REST OF THE CODE (Question 2 Part 3)---
    if np.abs(angle) < 1e-10:
        return np.eye(3)
    
    axis = axis / np.linalg.norm(axis) # Rounding error may make the axis grow, this line prevents that
    
    cos = np.cos(angle)
    sin = np.sin(angle)
    t = 1 - cos

    R = np.array([
        [t*axis[0]*axis[0] + cos,         t*axis[0]*axis[1] - axis[2]*sin, t*axis[0]*axis[2] + axis[1]*sin],
        [t*axis[0]*axis[1] + axis[2]*sin, t*axis[1]*axis[1] + cos,         t*axis[1]*axis[2] - axis[0]*sin],
        [t*axis[0]*axis[2] - axis[1]*sin, t*axis[1]*axis[2] + axis[0]*sin, t*axis[2]*axis[2] + cos    ]
    ])


    return R

def sample_grasps(
    center_point: np.ndarray,
    num_grasps: int,
    offset: float = 0.05,
) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates multiple random grasp poses around a given point cloud.

    Args:
        center: Center around which to sample grasps.
        num_grasps: Number of random grasp poses to generate
        offset: Maximum distance offset from the center (meters)

    Returns:
        list: List of rotations and Translations
    """

    grasp_poses_list = []
    for idx in range(num_grasps):
        # Sample a grasp center and rotation of the grasp
        # Sample a random vector in R3 for axis angle representation
        # Return the rotation as rotation matrix + translation
        # Translation implies translation from a center point
        ############################TODO############################
        # Sample a grasp center by adding random offsets to the center point
        grasp_center = center_point + np.random.uniform(-offset, offset, size=3)
 
        # Sample a random axis-angle representation
        random_axis = np.random.uniform(-1, 1, size=3)
        random_axis /= np.linalg.norm(random_axis)  # Normalize the axis
        random_angle = np.random.uniform(0, 2 * np.pi)  # Random angle in radians
        
        # Convert axis-angle to rotation matrix
        R = exp_map_so3(random_axis, random_angle)

        ######################################################
        assert R.shape == (3, 3)
        assert grasp_center.shape == (3,)
        grasp_poses_list.append((R, grasp_center))

    return grasp_poses_list

def check_grasp_collision(
    grasp_meshes: Sequence[o3d.geometry.TriangleMesh],
    object_mesh: o3d.geometry.TriangleMesh,
    num_colisions: int = 5,
    tolerance: float = 0.0001) -> bool:
    """
    Checks for collisions between a gripper grasp pose and target object
    using point cloud sampling.

    Args:
        grasp_meshes: List of mesh geometries representing the gripper components
        object_mesh: Triangle mesh of the target object
        num_collisions: Threshold on how many points to check
        tolerance: Distance threshold for considering a collision (in meters)

    Returns:
        bool: True if collision detected between gripper and object, False otherwise
    """
    # Combine gripper meshes
    combined_gripper = o3d.geometry.TriangleMesh()
    for mesh in grasp_meshes[1:]:
        combined_gripper += mesh

    # Sample points from both meshes
    num_points = 1000 # Subsample both meshes to this many points
    #######################TODO#######################
    point_cloud_object = object_mesh.sample_points_uniformly(number_of_points=num_points) #Sample point cloud of object
    
    point_cloud_grasp = o3d.geometry.PointCloud()
    for mesh in grasp_meshes:
        point_cloud_grasp += mesh.sample_points_uniformly(number_of_points=int(num_points/4)) #Sample point cloud of grasp
    

    ##################################################
    # Build KDTree for object points
    is_collision = False
    #######################TODO#######################
    kdtree = o3d.geometry.KDTreeFlann(point_cloud_object) # create KDTree
    for query_point in point_cloud_grasp.points:
        
        [k, idx, dist] = kdtree.search_knn_vector_3d(query_point, num_colisions) # Search points close and get the distance
        for distance in dist:
            if distance < tolerance:
                is_collision = True


    #######################TODO#######################

    return is_collision

# Fuction used to visualize multiple objects
def visualize_3d_objs(objs: Sequence[Any]) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Output viz')
    for obj in objs:
        vis.add_geometry(obj)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, 1])
    ctr.set_up([0, 1, 0])
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


def grasp_dist_filter(center_grasp: np.ndarray,
                      mesh_center: np.ndarray,
                      tolerance: float = 0.05)->bool:
    is_within_range = False
    #######################TODO#######################
    distance = np.linalg.norm(center_grasp - mesh_center) #Verify distance between centers
    if distance < tolerance:
        is_within_range = True

    ##################################################
    return is_within_range

### Auxiliar function: convert lineset to pointcloud
def lineset_to_pointcloud(lineset, points_per_line=1000):
    points = np.asarray(lineset.points)
    lines = np.asarray(lineset.lines)

    point_list = []
    for line in lines:
        start_point = points[line[0]]
        end_point = points[line[1]]
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        spacing = length / points_per_line
        direction = direction / length
        num_points = int(length / spacing) + 1
        for i in range(num_points):
            point = start_point + i * spacing * direction
            point_list.append(point)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_list)
    
    return point_cloud

def check_grasp_containment(
    left_finger_center: np.ndarray,
    right_finger_center: np.ndarray,
    finger_length: float,
    object_pcd: o3d.geometry.PointCloud,
    num_rays: int,
    rotation_matrix: np.ndarray, # rotati
) -> Tuple[bool, float]:
    """
    Checks if any line between the gripper fingers intersects with the object mesh.

    Args:
        left_finger_center: Center of Left finger of grasp
        right_finger_center: Center of Right finger of grasp
        finger_length: Finger Length of the gripper.
        object_pcd: Point Cloud of the target object
        clearance_threshold: Minimum required clearance between object and gripper

    Returns:
        tuple[bool, float]: (intersection_exists, intersection_depth)
        - intersection_exists: True if any line between fingers intersects object
        - intersection_depth: Depth of deepest intersection point
    """

    left_center = np.asarray(left_finger_center)
    right_center = np.asarray(right_finger_center)

    intersections = []
    
    # Check for intersections between corresponding points
    object_tree = o3d.geometry.KDTreeFlann(object_pcd)

    #######################TODO#######################

    contained = False

    #Create the Rays between the fingers using lineset
    instances = np.linspace(-0.5, 0.5, num_rays)
    left_points = [left_center + (rotation_matrix @ np.array([0,finger_length,0]))* i for i in instances]
    right_points = [right_center + (rotation_matrix @ np.array([0,finger_length,0]))* i for i in instances]

    
    rays_pcds = []
    for j in range(0, num_rays):
        ray_set = o3d.geometry.LineSet()
        ray_set.points = o3d.utility.Vector3dVector([left_points[j], right_points[j]])
        ray_set.lines = o3d.utility.Vector2iVector([[0,1]])
        #Convert each line into a pointcloud with 1000 points each
        rays_pcds.append(lineset_to_pointcloud(ray_set,1000))
    

    #Check for intersection (colision between rays and object)
    #For each ray
    for ray in rays_pcds:
        #Use the same method as before to check collisions
        for point in np.asarray(ray.points):
            # searc the nearest neighbor of the point in the KDTree
            [k, idx, distances] = object_tree.search_knn_vector_3d(point, 1)
            #If a collision is detected
            if k > 0 and distances[0] <= 0.0001:  
                #Consider that an occurance of intersection for one ray
                intersections.append(ray)
                contained = True
                break
    
    
    containment_ratio = len(intersections)/(num_rays)
    return contained, containment_ratio