import numpy as np


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
            rgb: RGBA image as numpy array [width, height, 4]
            depth: Depth map as numpy array [width, height]
            seg: Segmentation map as numpy array [width, height]
    """
    # RGBA - channel Range [0-255]
    rgb = np.reshape(rgbpx, [width, height, 4])
    # Depth Map Range [0.0-1.0]
    depth = np.reshape(depthpx, [width, height])
    # Segmentation Map Range {obj_ids}
    seg = np.reshape(segpx, [width, height])

    return rgb, depth, seg
