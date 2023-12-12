from typing import Tuple

import numpy as np
import cv2


def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for Braitenberg-like control
                            using the masked left lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    width = shape[1]
    steer_matrix_left = np.ones(shape)
    steer_matrix_left[:,int(np.floor(width/2)):width + 1] = 0
    # ---
    return steer_matrix_left


def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for Braitenberg-like control
                             using the masked right lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    width = shape[1]
    steer_matrix_right = np.ones(shape)
    steer_matrix_right[:,0:int(np.floor(width/2))] = 0
    # ---
    return steer_matrix_right


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = image.shape

    # TODO: implement your own solution here
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert the image to HSV for any color-based filtering
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # The image-to-ground homography associated with this image
    H = np.array([-4.137917960301845e-05, -0.00011445854191468058, -0.1595567007347241, 
                0.0008382870319844166, -4.141689222457687e-05, -0.2518201638170328, 
                -0.00023561657746150284, -0.005370140574116084, 0.9999999999999999])

    H = np.reshape(H,(3, 3))
    Hinv = np.linalg.inv(H)

    mask_ground = np.ones(img.shape, dtype=np.uint8)

    # Iterate through each pixel in the mask
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # Transform pixel coordinates using Hinv
            transformed_coords = np.dot(Hinv, np.array([x, y, 1]))
            # Normalize the coordinates
            normalized_coords = transformed_coords / transformed_coords[2]
            # Use x-coordinate for checking against the horizon
            if normalized_coords[0] > 170:
                mask_ground[y, x] = 0

    sigma = 5
    # Smooth the image using a Gaussian kernel
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)

    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)

    # Compute the orientation of the gradients
    Gdir = cv2.phase(np.array(sobelx, np.float32), np.array(sobely, dtype=np.float32), angleInDegrees=True)

    threshold = 50
    mask_mag = (Gmag > threshold)

    tolerance_white = 100
    targe_white = np.array([160, 0, 240])
    white_lower_hsv = np.array([max(0, targe_white[0] - tolerance_white), max(0, targe_white[1] - tolerance_white), max(0, targe_white[2] - tolerance_white)])
    white_upper_hsv = np.array([min(179, targe_white[0] + tolerance_white), min(255, targe_white[1] + tolerance_white), min(255, targe_white[2] + tolerance_white)])
    tolerance_yellow = 150
    target_yellow = np.array([38, 240, 98])
    yellow_lower_hsv = np.array([max(0, target_yellow[0] - tolerance_yellow), max(0, target_yellow[1] - tolerance_yellow), max(0, target_yellow[2] - tolerance_yellow)])
    yellow_upper_hsv = np.array([min(179, target_yellow[0] + tolerance_yellow), min(255, target_yellow[1] + tolerance_yellow), min(255, target_yellow[2] + tolerance_yellow)])

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

    mask_left = get_steer_matrix_left_lane_markings([h,w])
    mask_right = get_steer_matrix_right_lane_markings([h,w])
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)
    mask_left_edge = mask_ground * mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_ground * mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return mask_left_edge, mask_right_edge
