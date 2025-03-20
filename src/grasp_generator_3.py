import numpy as np
import open3d as o3d
import numpy as np
import open3d as o3d
from typing import Any, Sequence, Optional
from scipy.spatial import KDTree
from src.create_grabber import *
from src.robot import *
from scipy.spatial.transform import Rotation as R
from giga.perception import *
from giga.grasp_sampler import GpgGraspSamplerPcl
import cv2

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

# Uses the segmentation camera to aproximate the object to a point_cloud
def recreate_3d_system_object(grayscale, depth, center_x, center_y, projection_matrix, stat_viewMat, z_plane = 1.25, radius=75, num_points=10000):
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

    # Project all points onto this plane
    projected_points = points.copy()
    projected_points[:, 2] = z_plane  # Set all Z coordinates to the plane level

    Intermediary_points = np.vstack((points, projected_points))

    pcd_Intermediary = o3d.geometry.PointCloud()
    pcd_Intermediary.points = o3d.utility.Vector3dVector(Intermediary_points)

    # Correct point cloud by adding exta points
    # Compute a mesh using the alpha shape algorithm
    alpha = 0.1  # Adjust this value based on your dataset
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_Intermediary, alpha)

    Final_points = np.asarray(mesh.sample_points_poisson_disk(number_of_points=250).points)
    Final_points = Final_points[Final_points[:, 2] >= z_plane+0.005]
    Final_points = Final_points - [0, 0, z_plane+0.005]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Final_points)

    # Visualize the point cloud
    #o3d.visualization.draw_geometries([pcd])
    return pcd



def grasp_point_cloud_2(grayscale, depth, projection_matrix, stat_viewMat):
    if True:
        center_x, center_y = center_object(grayscale)

        pcd_object = recreate_3d_system_object(grayscale, depth, center_x, center_y, projection_matrix, stat_viewMat)  

        bbox = pcd_object.get_axis_aligned_bounding_box()
        pcd_object = pcd_object.crop(bbox)

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    
        vis_meshes = [pcd_object, coordinate_frame]

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd_object, coordinate_frame])

        # Grasp Sampling using GPD
        num_parallel_workers = 2
        num_grasps = 15

        sampler = GpgGraspSamplerPcl(0.05-0.0075)
        safety_dist_above_table = 0.005  # Adjusted safety distance
        grasps, grasps_pos, grasps_quat = sampler.sample_grasps_parallel(pcd_object, num_parallel=num_parallel_workers, num_grasps=num_grasps, max_num_samples=80,
                                    safety_dis_above_table=safety_dist_above_table, show_final_grasps=False)  # Enable final grasp visualization
        
        all_grasp_meshes = []
        for grasp_position, grasp_quaternion in zip(grasps_pos, grasps_quat):
            print(grasp_position, grasp_quaternion)
            grasp_rotation = R.from_quat(grasp_quaternion).as_matrix()
            grasp = create_grasp_mesh(center_point=grasp_position, rotation_matrix=grasp_rotation)
            all_grasp_meshes.append(grasp)
            vis_meshes.extend(grasp)

        visualize_3d_objs(vis_meshes)