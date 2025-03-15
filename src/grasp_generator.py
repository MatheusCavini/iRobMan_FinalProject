import numpy as np
import cv2

#[DK:2025-02-26] Grasp angle finder
############################################################################
def find_grasp_angle(image):

    _, binary = cv2.threshold(image, 100, 200, cv2.THRESH_BINARY_INV)

    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("No object found.")
        return None
    
    # Get the largest contour (assuming it's the object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit a rotated rectangle to the contour
    rect = cv2.minAreaRect(largest_contour)
    
    # Extract center and angle
    (cx, cy), (width, height), angle = rect
    
    # Adjust angle for grasping (ensuring minimal rotation)
    if width < height:
        angle += 90
    
    
    return (cx, cy, angle)

def euclidean_distance(first_vector0, first_vector1, second_vector0, second_vector1):
    return ((first_vector0 - second_vector0)**2 + ((first_vector1 - second_vector1)**2))**0.5

def verify_location_threshhold(img_seg, location_threshhold):
    center_x, center_y, grasp_angle = find_grasp_angle(img_seg)
    center_image_x = len(img_seg[0])/2
    center_image_y = len(img_seg)/2
    return euclidean_distance(center_x, center_y, center_image_x, center_image_y) < location_threshhold



def grasping_generator(img_seg):
    center_x, center_y, grasp_angle = find_grasp_angle(img_seg)
    center_image_x = len(img_seg[0])/2
    center_image_y = len(img_seg)/2
    correction_angle = np.arctan((center_y-center_image_y)/(center_x-center_image_x))
    return grasp_angle, correction_angle

'''The function can also be used to get the center pixel of the object. This is useful for delicate adjustments. From what we tested,
   the global camera estimate appears to be accurate enough to capture the object. Therefore, we have not implemented this functionality fully'''
############################################################################

import numpy as np
import open3d as o3d
import numpy as np
import open3d as o3d
from typing import Any, Sequence, Optional
from src.create_grabber import *
import copy


# Uses the segmentation camera to aproximate the object to a point_cloud
def recreate_3d_system_object(grayscale, depth, center_x, center_y, projection_matrix, stat_viewMat, radius=100, num_points=10000):
    """
    Recreates a 3D system from grayscale intensity and depth information, selecting random points around a specified pixel.
    
    :param grayscale: 2D numpy array representing grayscale values
    :param depth: 2D numpy array representing distance from the camera
    :param center_x: X coordinate of the center pixel
    :param center_y: Y coordinate of the center pixel
    :param radius: Radius around the center pixel to select points
    :param num_points: Number of random points to sample
    """
    h, w = grayscale.shape
    
    # Create a mask for points around the specified center
    y_min, y_max = max(0, center_y - radius), min(h, center_y + radius)
    x_min, x_max = max(0, center_x - radius), min(w, center_x + radius)
    
    # Extract relevant points
    y_coords, x_coords = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
    y_coords, x_coords = y_coords.flatten(), x_coords.flatten()
    z = depth[y_coords, x_coords]
    colors = grayscale[y_coords, x_coords]
    
    # Filter out points where z < 2.5 and keep only white pixels (grayscale value close to 1.0)
    mask = (colors == 44)
    x_coords, y_coords, z, colors = x_coords[mask], y_coords[mask], z[mask], colors[mask]
  
    # Randomly select num_points from the valid points
    if len(x_coords) > num_points:
        indices = np.random.choice(len(x_coords), num_points, replace=False)
        x_coords, y_coords, z, colors = x_coords[indices], y_coords[indices], z[indices], colors[indices]
    
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
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.stack((colors, colors, colors), axis=1))
    pcd.paint_uniform_color([1, 0.706, 0])

    # Correct point cloud by adding exta points
    # Compute a mesh using the alpha shape algorithm
    alpha = 0.1  # Adjust this value based on your dataset
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    
    # Visualize the point cloud
    #o3d.visualization.draw_geometries([pcd_filled])
    return mesh

# Uses the segmentation camera to aproximate the table to a point_cloud
def recreate_3d_system_table(grayscale, depth, center_x, center_y, projection_matrix, stat_viewMat, radius=100, num_points=10000):
    """
    Recreates a 3D system from grayscale intensity and depth information, selecting random points around a specified pixel.
    
    :param grayscale: 2D numpy array representing grayscale values
    :param depth: 2D numpy array representing distance from the camera
    :param center_x: X coordinate of the center pixel
    :param center_y: Y coordinate of the center pixel
    :param radius: Radius around the center pixel to select points
    :param num_points: Number of random points to sample
    """
    h, w = grayscale.shape
    
    # Create a mask for points around the specified center
    y_min, y_max = max(0, center_y - radius), min(h, center_y + radius)
    x_min, x_max = max(0, center_x - radius), min(w, center_x + radius)
    
    # Extract relevant points
    y_coords, x_coords = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
    y_coords, x_coords = y_coords.flatten(), x_coords.flatten()
    z = depth[y_coords, x_coords]
    colors = grayscale[y_coords, x_coords]
    
    # Filter out points where z < 2.5 and keep only white pixels (grayscale value close to 1.0)
    mask = (colors == 120)
    x_coords, y_coords, z, colors = x_coords[mask], y_coords[mask], z[mask], colors[mask]
    print(x_coords)
    
    # Randomly select num_points from the valid points
    if len(x_coords) > num_points:
        indices = np.random.choice(len(x_coords), num_points, replace=False)
        x_coords, y_coords, z, colors = x_coords[indices], y_coords[indices], z[indices], colors[indices]
    
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
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.stack((colors, colors, colors), axis=1))
    pcd.paint_uniform_color([1, 0, 0.706])

    # Correct point cloud by adding exta points
    # Compute a mesh using the alpha shape algorithm
    alpha = 99  # Adjust this value based on your dataset
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    
    # Visualize the point cloud
    #o3d.visualization.draw_geometries([mesh])
    return mesh

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

#Calculate the center of the object using the camera
def center_object(img_seg):

    #output_image = cv2.cvtColor(img_seg, cv2.COLOR_GRAY2BGR)

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
            '''cv2.circle(output_image, (cx, cy), 5, (255, 0, 0), -1)  # Azul
    
    cv2.imshow("Detected Object", output_image)
    cv2.waitKey(1)'''

    return cx, cy


#Generates multiple grasos randomly
def generate_multiple_grasps(center, pcd_object):
    sample_grasp_lists = sample_grasps(center_point=center, num_grasps=100)

    #######################TODO######################
    all_grasp_meshes = []
    for R_matrix, grasp_center in sample_grasp_lists:

        grasp_mesh = create_grasp_mesh(center, rotation_matrix=R_matrix) # create a new mesh for another grasp orientation

        # Store the transformed mesh
        all_grasp_meshes.append(grasp_mesh)
    
    '''vis_meshes = [pcd_object]
    for grasp_mesh in all_grasp_meshes:
        vis_meshes.extend(grasp_mesh)
    
    visualize_3d_objs(vis_meshes)'''
    return all_grasp_meshes

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

def check_grasp_collision(
    grasp_meshes: Sequence[o3d.geometry.TriangleMesh],
    object_mesh: o3d.geometry.TriangleMesh,
    num_colisions: int = 10,
    tolerance: float = 0.00001) -> bool:
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
    num_points = 5000 # Subsample both meshes to this many points
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

def merge_meshes(mesh_object, mesh_table):
    pcd_object = mesh_object.sample_points_uniformly(number_of_points=5000)
    pcd_table =  mesh_table.sample_points_uniformly(number_of_points=10000)
    merged_pcd = o3d.geometry.PointCloud()
    
    # Merge points
    merged_points = np.vstack((np.asarray(pcd_object.points), np.asarray(pcd_table.points)))
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)

    if pcd_object.has_colors() and pcd_table.has_colors():
        merged_colors = np.vstack((np.asarray(pcd_table.colors), np.asarray(pcd_object.colors)))
        merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    
    alpha = 0.03  # Adjust alpha for tight or loose fitting
    mesh_merged = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(merged_pcd, alpha)
    return mesh_merged


# Takes the grayscale and depth image and returns optimal grasp position and orientation
def grasp_point_cloud(grayscale, depth, projection_matrix, stat_viewMat):
    cv2.imwrite("Gray_SCALE.png", grayscale)
    center_x, center_y = center_object(grayscale)
   
    mesh_object = recreate_3d_system_object(grayscale, depth, center_x, center_y, projection_matrix, stat_viewMat)
    mesh_table = recreate_3d_system_table(grayscale, depth, center_x, center_y, projection_matrix, stat_viewMat)
    
    points = np.asarray(mesh_object.sample_points_uniformly(number_of_points=5000).points)
    center = points.mean(axis=0)
    all_grasp_meshes = generate_multiple_grasps(center, mesh_object)


   
    vis_meshes = [mesh_object, mesh_table]
    mesh_object_plus_table = merge_meshes(mesh_object, mesh_table)
    #visualize_3d_objs(vis_meshes)
    for grasp_mesh in all_grasp_meshes:
        if not check_grasp_collision(grasp_mesh, mesh_object_plus_table):
            vis_meshes.extend(grasp_mesh)

    visualize_3d_objs(vis_meshes)


    return
