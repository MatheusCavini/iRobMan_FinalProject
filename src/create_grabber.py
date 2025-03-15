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
    gripper_distance: float = 0.1,
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
        grasp_center = center_point
 
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