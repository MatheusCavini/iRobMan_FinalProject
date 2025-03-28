import numpy as np
from giga.perception import *
from giga.utils.transform import Rotation, Transform
from giga.utils.implicit import as_mesh
from giga.grasp_sampler import GpgGraspSamplerPcl
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import trimesh
import cv2
from typing import Any, Sequence
from src.robot import Robot



###[DK: 2025-03-15] create methods for isolating the objects###
###########################################################

# Fuction used to visualize multiple objects, taken from the Assignment 2
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

#Returns a estimation for a pixel in the static camera 
def point_3D_estimator(object_info, depth_real, projection_matrix, view_matrix):
    if object_info[0] == 0 and object_info[1] == 0:
        return np.array([0, 0, 0])
    
    x, y = object_info
    projection_matrix = np.array(projection_matrix).reshape(4,4)
    view_matrix = np.array(view_matrix).reshape(4, 4).T

    h, w = depth_real.shape
    
    #Extract PyBullet projection matrix parameters
    fx = projection_matrix[0, 0] * w / 2
    fy = projection_matrix[1, 1] * h / 2
    cx = (1 - projection_matrix[0, 2]) * w / 2
    cy = (1 + projection_matrix[1, 2]) * h / 2

    #Get depth value for obstacle center at (x, y)
    depth_value = depth_real[int(y), int(x)]

    #Convert to camera coordinates
    Z = depth_value
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    
    #Create homogeneous coordinate vector
    P_cam = np.array([X, Y, Z]).reshape(-1, 3)
    P_cam_homogeneous = np.append(P_cam, 1).reshape(4, 1)

    #Define a camera transfomation matrix
    transformation_matrix = np.array([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])

    #Apply the transformation
    P_cam_transformed = np.dot(transformation_matrix, P_cam_homogeneous)
    P_cam_transformed = P_cam_transformed.reshape(4,1)
    
    #Compute the inverse view matrix
    V_inv = np.linalg.inv(view_matrix)
    
    #Transform to world coordinates
    P_world = (V_inv @ P_cam_transformed)
    
    # Return only (x, y, z) world coordinates
    return (P_world[:3]/P_world[3]).flatten()



#Calculate the center of the object using the camera
def center_object(img_seg):
    '''
    Calculate the center of the object using the segmentation image.

    img_seg: 2D numpy array representing the segmentation image.

    Returns:
        cx: x-coordinate of the center of the object.
        cy: y-coordinate of the center of the object.
    
    '''

     # Creating a mask with the segmentation values
    mask = (img_seg== 44).astype(np.uint8) * 255
    
    # Find the portions with the right gray tone
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx = 0
    cy = 0

    # Calculating the central point
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
    return cx, cy

# Uses the segmentation camera to aproximate the object to a point_cloud
def recreate_3d_system_object(grayscale, depth, projection_matrix, stat_viewMat,  alpha, z_plane = 1.25):
    """
    Recreates a 3D object from grayscale intensity and depth information.
    
    grayscale: 2D numpy array representing grayscale values from the static camera
    depth: 2D numpy array representing distance from the static camera
    projection_matrix: 3D numpy array representing the projection matrix (Intrinsic camera parameters)
    stat_viewMat: 3D numpy array representing the static camera view matrix (Intrinsic camera parameters)
    z_plane: float representing the level Z coordinate of the table plane

    Returns:
        pcd: Open3D point cloud object
        mesh: Open3D mesh object
    """
    
    y_coords, x_coords = np.where(grayscale == 44)  # Find the coordinates of the object in the grayscale image

    # Convert pixel coordinates to real-world 3D coordinates
    x_real = []
    y_real = []
    z_real = []
    for i in range(len(x_coords)):
        x, y, z = point_3D_estimator([x_coords[i], y_coords[i]], depth, projection_matrix, stat_viewMat)
        x_real.append(x)
        y_real.append(y)
        z_real.append(z) 
    
    # Create Open3D point cloud
    points = np.vstack((x_real, y_real, z_real)).T

    # Project all points onto the table plane
    projected_points = points.copy()
    projected_points[:, 2] = z_plane  # Set all Z coordinates to the plane level
    points = points - np.array([0, 0, z_plane])  # Translate original points to the origin
    projected_points = projected_points - np.array([0, 0, z_plane])  # Translate projected points to the origin

    Intermediary_points = np.vstack((points, projected_points)) # Combine original and projected points

    pcd_Intermediary = o3d.geometry.PointCloud()
    pcd_Intermediary.points = o3d.utility.Vector3dVector(Intermediary_points) #Create initial point cloud

    # Correct point cloud to make it smoother
    # Compute a mesh using the alpha shape algorithm
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_Intermediary, alpha)

    pcd = mesh.sample_points_poisson_disk(number_of_points=5000) # Sample points from the mesh to create a uniform point cloud
    pcd.paint_uniform_color([1, 0.706, 0]) # Set color to yellow 
   
    return pcd, mesh

###########################################################


###[DK: 2025-03-25] create methods for grasping generation###
###########################################################

# Function to create a mesh for the grasping fingers and palm (taken from the GIGA library)
def grasp2mesh(grasp, score, finger_depth=0.05):
    # color = cmap(float(score))
    # color = (np.array(color) * 255).astype(np.uint8)
    color = np.array([0, 250, 0, 180]).astype(np.uint8)
    radius = 0.1 * finger_depth
    w, d = grasp.width, finger_depth
    scene = trimesh.Scene()
    # left finger
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, -w / 2, d / 2])
    scale = [radius, radius, d]
    left_finger = trimesh.creation.cylinder(radius,
                                            d,
                                            transform=pose.as_matrix())
    scene.add_geometry(left_finger, 'left_finger')

    # right finger
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, w / 2, d / 2])
    scale = [radius, radius, d]
    right_finger = trimesh.creation.cylinder(radius,
                                             d,
                                             transform=pose.as_matrix())
    scene.add_geometry(right_finger, 'right_finger')

    # wrist
    pose = grasp.pose * Transform(Rotation.identity(), [0.0, 0.0, -d / 4])
    scale = [radius, radius, d / 2]
    wrist = trimesh.creation.cylinder(radius,
                                      d / 2,
                                      transform=pose.as_matrix())
    scene.add_geometry(wrist, 'wrist')

    # palm
    pose = grasp.pose * Transform(
        Rotation.from_rotvec(np.pi / 2 * np.r_[1.0, 0.0, 0.0]),
        [0.0, 0.0, 0.0])
    scale = [radius, radius, w]
    palm = trimesh.creation.cylinder(radius, w, transform=pose.as_matrix())
    scene.add_geometry(palm, 'palm')
    scene = as_mesh(scene)
    colors = np.repeat(color[np.newaxis, :], len(scene.faces), axis=0)
    scene.visual.face_colors = colors
    return scene


def sort_orientations_by_verticality(poses):
    """
    Sorts a list of grasping poses by how vertical the orientation is.

    poses (list): List of pairs of position and orientation (quaternion).

    Returns:
        List of sorted poses (more vertical first)
    """
    
    def vertical_score(pose):
        quat = pose[1]  # get [qx, qy, qz, qw]
        r = R.from_quat(quat)
        z_axis = r.apply([0, 0, -1])  # get transformed z-axis
        return z_axis[2]  # the vertical score
    
    # Sort in descending order
    return sorted(poses, key=vertical_score, reverse=True)

## Function to generate grasp points from the point cloud using GPD
def grasp_point_cloud_2(grayscale, depth, projection_matrix, stat_viewMat, Robot, alpha = 0.3):
    if True:

        pcd_object, _ = recreate_3d_system_object(grayscale, depth, projection_matrix, stat_viewMat, alpha) # Create point cloud from the static camera

        bbox = pcd_object.get_axis_aligned_bounding_box() ## Get the bounding box of the point cloud
        pcd_object = pcd_object.crop(bbox) # Crop the point cloud to the bounding box

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0]) # Create a coordinate frame for visualization
    
        vis_meshes = [pcd_object, coordinate_frame] # Initialize the list of PCD to visualize

        # Grasp Sampling using GPD
        num_parallel_workers = 2
        num_grasps = 30 # Number of grasps to sample

        sampler = GpgGraspSamplerPcl(0.05-0.0075)
        safety_dist_above_table = 0.005  # Adjusted safety distance
        grasps, grasps_pos, grasps_quat = sampler.sample_grasps_parallel(pcd_object, num_parallel=num_parallel_workers, num_grasps=num_grasps, max_num_samples=80,
                                    safety_dis_above_table=safety_dist_above_table, show_final_grasps=False)  # Sample grasps from the point cloud (PCD, position, quaternion orientation)
        
        available_grasps = []
        
        #transform the grasps created to point clouds
        grasp_mesh_list = [grasp2mesh(g, score=1) for g in grasps] # Create trimesh for the grasps
        for grasp_mesh, grasp_pos, grasp_quat in zip(grasp_mesh_list, grasps_pos, grasps_quat):
            points, _ = trimesh.sample.sample_surface(as_mesh(grasp_mesh), 5000) # Sample points from the grasp trimesh
           
            pcd_grasp = o3d.geometry.PointCloud()
            pcd_grasp.points = o3d.utility.Vector3dVector(points) # Create Open3D point cloud from the sampled points

            #if Robot.is_pose_reachable(grasp_pos, grasp_quat):
            #    pcd_grasp.paint_uniform_color([0, 1, 0])

            vis_meshes.append(pcd_grasp) # Add the grasp point cloud to the list of meshes to visualize
            available_grasps.append([grasp_pos + np.array([0, 0, 1.25]), grasp_quat]) # Add the grasp position and quaternion to the list of available grasps

        available_grasps = sort_orientations_by_verticality(available_grasps) # Sort the grasps by verticality

        visualize_3d_objs(vis_meshes) # Visualize the point cloud and grasps

        #If no available grasps were found, the reconstruction might have been bad. Run the function again ajusting the recontruction
        if len(available_grasps) == 0:
            print("Value of alpha: ", alpha)
            return grasp_point_cloud_2(grayscale, depth, projection_matrix, stat_viewMat, Robot, alpha = alpha - 0.1) 
        else:
            return available_grasps[0] # Return the first grasp (most vertical) as the best grasp point
###########################################################