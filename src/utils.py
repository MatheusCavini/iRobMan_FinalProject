import numpy as np
import pybullet as p


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
