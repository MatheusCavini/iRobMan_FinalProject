import cv2 as cv
import cv2
import numpy as np

def center_object(img_rgb, img_seg):

    img_rgb = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
   

    initial_points = np.float32([[119, 144], [520, 144], [0, 463], [639, 463]])

    final_points = np.float32([[0, 0], [639, 0], [0, 463], [639, 463]])

    w = cv.getPerspectiveTransform(initial_points, final_points)
    res_seg = cv.warpPerspective(img_seg,w,(img_seg.shape[1], img_seg.shape[0])) 
    res_rgb = cv.warpPerspective(img_rgb,w,(img_seg.shape[1], img_seg.shape[0])) 

    _, thresholded = cv2.threshold(res_seg, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def is_red(rgb):
        R, G, B = rgb
        return R == 41 and G == 41 and B == 205

    def is_brown(rgb):
        R, G, B = rgb
        return R == 79 and G == 99 and B == 112

    for contour in contours:
        M = cv2.moments(contour)
        
        if M["m00"] != 0:  
            cx = int(M["m10"] / M["m00"])  
            cy = int(M["m01"] / M["m00"])  

            if not is_brown(res_rgb[cy, cx]) and not is_red(res_rgb[cy, cx]):

                center_table = np.array([319.5, 105.5])
                distance_center = np.array([cy, cx]) - center_table

                distance_x = -1.3*distance_center[1]/639
                distance_y = 0.8*distance_center[0]/639

                return distance_x, distance_y
    return None