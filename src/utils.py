import numpy as np
import pybullet as p
import yaml
from scipy.spatial.transform import Rotation as R
import scipy

def pb_image_to_numpy(rgbpx, depthpx, segpx, width, height):
    """
    Convert pybullet camera images to numpy arrays.

    Args:
        rgbpx: RGBA pixel values
        depthpx: Depth map pixel values
        segpx: Segmentation map pixel values
        width: Image width
        height: Image height

    Returns:
        Tuple of:
            rgb: RGBA image as numpy array [height, width, 4]
            depth: Depth map as numpy array [height, width]
            seg: Segmentation map as numpy array [height, width]
    """
    # RGBA - channel Range [0-255]
    rgb = np.reshape(rgbpx, [height, width, 4])
    # Depth Map Range [0.0-1.0]
    depth = np.reshape(depthpx, [height, width])
    # Segmentation Map Range {obj_ids}
    seg = np.reshape(segpx, [height, width])

    return rgb, depth, seg

###[MC:2025-02-15] Quaternions and Vector-Angle conversion
###########################################################
def quaternion_to_axis_angle(q):
    q = np.array(q)  
    q = q / np.linalg.norm(q)  #normalize quaternion

    w = q[3]  
    v = q[:3]  

    angle = 2 * np.arccos(w)  #Get angle from w componenet

    if np.linalg.norm(v) < 1e-6:
        return np.array([0, 0, 0]), 0  #avoid 0 division for small angles

    axis = v / np.linalg.norm(v)  #normalize axis

    return axis, angle

def axis_angle_to_quaternion(axis, angle):
    axis = np.array(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)  #normalize axis
    
    half_angle = angle / 2.0
    sin_half_angle = np.sin(half_angle)
    
    w = np.cos(half_angle)
    x, y, z = axis * sin_half_angle
    
    return np.array([x, y, z, w])

def concatenate_quaternions(q1, q2):
    #generates the quaternion corresponding to 2 sequential rotations
    return p.multiplyTransforms([0, 0, 0], q2, [0, 0, 0], q1)[1] 
###########################################################


##[MC:2025-02-15] Function to get real depth data using depth map provided by camera and its parameters
###########################################################
def real_depth(depth, near, far):
    #Near and Far are camera parameters
    depth_real = 2 * far * near / (far + near - (2 * depth - 1) * (far - near))
    return depth_real

def R_matrix_2_axisangle(R_matrix):
    # Convert to axis-angle representation
    rotation = R.from_matrix(R_matrix)
    axis_angle = rotation.as_rotvec()

    # Extract rotation axis and angle
    angle = np.linalg.norm(axis_angle)  # Angle in radians
    axis = axis_angle / angle if angle != 0 else np.array([0, 0, 1])  # Avoid division by zero
    return axis, angle
###########################################################

##[DK:2025-03-25] Function to obtain the direction vector for the pre-grasping
###########################################################
def quaternion_to_direction(quaternion):
    """
    Convert a quaternion to a direction vector.

    Parameters:
    - quaternion (list or numpy array): The quaternion [qx, qy, qz, qw].

    Returns:
    - numpy array: The rotated direction vector.
    """
    quaternion = np.array(quaternion, dtype=float)

    # Define the reference approach direction (e.g., Z-axis)
    reference_direction = np.array([0, 0, 1])

    # Convert quaternion to rotation object
    rotation = scipy.spatial.transform.Rotation.from_quat(quaternion)

    # Rotate the reference direction
    direction = rotation.apply(reference_direction)

    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)

    return direction
###########################################################

##[DK:2025-03-28] Simple array comparator
###########################################################
def compare_arrays(A, B):
    if len(A) != len(B):
        return False
    for i in range(len(A)):
        if A[i] != B[i]:
            return False
    return True
###########################################################