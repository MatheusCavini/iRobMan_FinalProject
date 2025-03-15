import cv2
import numpy as np
import filterpy.kalman as kf


##[MC:2025-02-15] Set of functions to measure 3D position of objects
###########################################################

def detect_obstacle_2D(image):
    #Convert the numpy image from camera to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    #Define the HSV values range for red (obstacle color)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    #Create binary masks for red color in image 
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    #Apply dilation and erosion step to decrease border irregularities
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    #Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(eroded, (9, 9), 2)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=10, maxRadius=0)

    #Initialize list to store circle information
    circle_info = [] #stores (x,y) coordinates of the center and radius r

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            circle_info.append((x, y, r))

    
    #DEBUG: Display the image with highlighted circles
    # Draw the circle in the original image in green
    circle_info = sorted(circle_info, key=lambda c: c[0], reverse=True)
    """for idx in range(len(circle_info)):
        x, y, r = circle_info[idx]
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        cv2.putText(image, str(idx), (x-2, y+2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Detected Red Circles", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)"""

    #return circle info ordered by y coordinate
    return circle_info


def obstacle_3D_estimator(circle_info, depth_real, projection_matrix, view_matrix, obstacle_idx):
    if len(circle_info) <= obstacle_idx:
        return np.array([0, 0, 0])
    
    x, y, r = circle_info[obstacle_idx]
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

    #Refine estimation including radius information
    x1 = (x+r - cx)* Z / fx
    x2 = (x-r - cx)* Z / fx
    R = np.abs((x1-x2))/2
    Z = Z + R
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

    
    ###########################################################



##[MC:2025-03-06] Create kalman filter for tracking obstacle positions
###########################################################
import numpy as np
from filterpy.kalman import KalmanFilter

class ObstacleKalmanFilter:
    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=1.0):
       
        self.dt = dt
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        # State vector: [x, y, z, vx, vy, vz]
        self.kf.x = np.zeros(6)

        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, dt, 0,  0],  # x' = x + vx*dt
            [0, 1, 0, 0, dt,  0],  # y' = y + vy*dt
            [0, 0, 1, 0,  0, dt],  # z' = z + vz*dt
            [0, 0, 0, 1,  0,  0],  # vx' = vx
            [0, 0, 0, 0,  1,  0],  # vy' = vy
            [0, 0, 0, 0,  0,  1],  # vz' = vz
        ])

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])

        
        self.kf.P *= 1000  

        # Process noise matrix
        q = process_noise
        self.kf.Q = np.array([
            [q, 0, 0, 0, 0, 0],
            [0, q, 0, 0, 0, 0],
            [0, 0, q, 0, 0, 0],
            [0, 0, 0, q, 0, 0],
            [0, 0, 0, 0, q, 0],
            [0, 0, 0, 0, 0, q],
        ])

        # Measurement noise matrix
        r = measurement_noise
        self.kf.R = np.array([
            [r, 0, 0],
            [0, r, 0],
            [0, 0, r],
        ])



    def update(self, measurement):
        
        self.kf.predict()
        self.kf.update(np.array(measurement))
        return self.kf.x[:3]
    
    def get_velocity(self):
        return self.kf.x[3:]


    ###########################################################