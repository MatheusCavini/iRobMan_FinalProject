import cv2 as cv
import cv2
import numpy as np

#[DK:2025-02-16] Object detection and coordinate estimation
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

    return np.array([cx, cy])


def object_3D_estimator(object_info, depth_real, projection_matrix, view_matrix):
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