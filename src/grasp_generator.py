import numpy as np
import cv2

#[DK:2025-02-26] Grasp angle finder
############################################################################
def find_grasp_angle(image):

    # Apply thresholding to segment the object
    cv2.imwrite("imagem_chegada.png", image)
    _, binary = cv2.threshold(image, 100, 200, cv2.THRESH_BINARY_INV)
    cv2.imwrite("binary.png", binary*60)

    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("No object found.")
        return None
    
    # Get the largest contour (assuming it's the object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit a rotated rectangle to the contour
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)  # Convert to integer points
    
    # Extract center and angle
    (cx, cy), (width, height), angle = rect
    
    # Adjust angle for grasping (ensuring minimal rotation)
    if width < height:
        angle += 90
    
    
    return (cx, cy, angle)


def grasping_generator(img_deth, img_seg):
    center_x, center_y, grasp_angle = find_grasp_angle(img_seg)
    #deth_to_objetc = np.max(img_deth)
    
    #position = current_position - [0, 0, deth_to_objetc]
    return grasp_angle 

'''The function can also be used to get the center pixel of the object. This is useful for delicate adjustments. From what we tested,
   the global camera estimate appears to be accurate enough to capture the object. Therefore, we have not implemented this functionality fully'''
############################################################################