import cv2
import time
import numpy as np

def detect_curve():
    frame = cv2.imread("problem1.png")

    start_time = time.time()

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of white color in HSV
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([255, 255, 255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Bitwise-AND mask and original image
    white_pixels = cv2.bitwise_and(frame, frame, mask=mask)



    # Display the resized image
    cv2.imshow("Resized White Pixels", white_pixels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_curve()
