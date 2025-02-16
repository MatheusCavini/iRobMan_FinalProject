import cv2 as cv
import cv2
import numpy as np

#[DK:2025-02-16] Object detection and coordinate estimation
def center_object(img_seg):

    initial_points = np.float32([[119, 144], [520, 144], [0, 463], [639, 463]])

    final_points = np.float32([[0, 0], [639, 0], [0, 463], [639, 463]])

    w = cv.getPerspectiveTransform(initial_points, final_points)
    res_seg = cv.warpPerspective(img_seg,w,(img_seg.shape[1], img_seg.shape[0]))  #Warping the perspective for a planned image

     # Creating a mask with the segmentation values
    mask = (res_seg== 44).astype(np.uint8) * 255
    
    # Find the portions with the right gray tone
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Debuging
    '''output_image = cv2.cvtColor(res_seg, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image, contours, -1, (0, 0, 255), 2)  '''

    distance_x = 5
    distance_y = 5

    # Calculating the central point
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            #cv2.circle(output_image, (cx, cy), 5, (255, 0, 0), -1)  # Azul

            # Converting the pixel in the planned image to global position
            center_table = np.array([319.5, 105.5])
            distance_center = np.array([cy, cx]) - center_table

            distance_x = -1.3*distance_center[1]/639
            distance_y = 0.8*distance_center[0]/639
            
        
    #cv2.imshow("Detected Object", output_image)
                
    #cv2.waitKey(1)

    return distance_x, distance_y