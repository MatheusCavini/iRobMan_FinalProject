
import numpy as np
import open3d as o3d
import numpy as np
import open3d as o3d
from src.create_grabber import *
import cv2


# Uses the segmentation camera to aproximate the object to a point_cloud
def recreate_3d_system_object(grayscale, depth, center_x, center_y, projection_matrix, stat_viewMat,z_plane = 1.25,  radius=75, num_points=10000):
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

    pcd = mesh.sample_points_poisson_disk(number_of_points=5000)
    pcd.paint_uniform_color([1, 0.706, 0])

   
    return pcd, mesh

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
    return cx, cy


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

def table_grasp_collision(grasp_pcd):
    for block in grasp_pcd:
        block = block.sample_points_poisson_disk(number_of_points=5000)
        points = np.asarray(block.points)
        min_value = np.min(points[:, 2])
        
        if min_value < 1.25:
            return True
    return False

# Takes the grayscale and depth image and returns optimal grasp position and orientation
def grasp_point_cloud_1(grayscale, depth, projection_matrix, stat_viewMat, offset=0.01):
    
    center_x, center_y = center_object(grayscale)
    
    if True:

        #Obtain the center of the object
        center_x, center_y = center_object(grayscale)
    
        #Recreate the 3D system
        object_pcd, object_mesh = recreate_3d_system_object(grayscale, depth, center_x, center_y, projection_matrix, stat_viewMat)
        o3d.io.write_point_cloud("output.pcd", object_pcd) 
        kdtree = o3d.geometry.KDTreeFlann(object_pcd)
        

        # Visualize the object, table, and grasp poses
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        vis_meshes = [object_pcd, coordinate_frame]

        #Recalculate the center of the object
        points = np.asarray(object_pcd.points)
        center = np.array(points.mean(axis=0)) + np.array([0, 0, 0.01])

        # Generate grasp poses
        sample_grasp_lists = sample_grasps(center_point=center, num_grasps=100, offset=offset)
        all_grasp_meshes = []
        for R_matrix, grasp_center in sample_grasp_lists:
            grasp_mesh = create_grasp_mesh(center_point= grasp_center, rotation_matrix=R_matrix) # create a new mesh for another grasp orientation
            all_grasp_meshes.append(grasp_mesh)# Store the transformed mesh
        
       

        # Filter grasps based on collision, distance and feasibility
        possible_grasps = []
        for pose, grasp_mesh in zip(sample_grasp_lists, all_grasp_meshes):
            not_collision = False
            contained = False
            contained_percentage = 0.0
            left_finger, right_finger = grasp_mesh[0], grasp_mesh[1]

            print(".")

            #Check the grasp containment
            contained, ratio = check_grasp_containment(
                left_finger_center=left_finger.get_center(),
                right_finger_center=right_finger.get_center(),
                finger_length=0.1,
                object_pcd=object_pcd,
                num_rays=50,
                rotation_matrix=pose[0]
            )

            if contained:
                not_collision = not check_grasp_collision(grasp_mesh, kdtree)
                if not_collision:
                    not_collision = not table_grasp_collision(grasp_mesh)
                    if not_collision:
                        #Check grasp collision and distance to object
                        contained_percentage = ratio
                        color = [1.0 - contained_percentage,
                                contained_percentage, 0.0]
                        for g_mesh in grasp_mesh:
                            # to make results look more interpretable
                            g_mesh.compute_vertex_normals()
                            g_mesh.paint_uniform_color(color)
                        vis_meshes.extend(grasp_mesh)                
                        possible_grasps.append(pose[0]) #Store the feasible grasps
        
        visualize_3d_objs(vis_meshes)
            
        if len(possible_grasps) == 0:
            print("No feasible grasps found.")
            return grasp_point_cloud_1(grayscale, depth, projection_matrix, stat_viewMat, offset=offset+0.01)
            
        if len(possible_grasps) == 1:
            return possible_grasps[0]
            
        else:
            return possible_grasps[0]