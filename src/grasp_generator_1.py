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

def grasping_generator(img_seg):
    center_x, center_y, grasp_angle = find_grasp_angle(img_seg)
    center_image_x = len(img_seg[0])/2
    center_image_y = len(img_seg)/2
    correction_angle = np.arctan((center_y-center_image_y)/(center_x-center_image_x))
    return grasp_angle, correction_angle

'''The function can also be used to get the center pixel of the object. This is useful for delicate adjustments. From what we tested,
   the global camera estimate appears to be accurate enough to capture the object. Therefore, we have not implemented this functionality fully'''