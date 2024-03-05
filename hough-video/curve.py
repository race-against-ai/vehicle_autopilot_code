import cv2
import time
import numpy as np

def canny(image):
    # Get canny image (remove blur and apply scale)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_of_interest(image):
    cropped_image = image[300:500, 250:850]
    return cropped_image


def detect_curve(frame):
    start_time = time.time()
    canny_image = canny(frame)
    canny_time = (time.time() - start_time) * 1000
    cropped_image = region_of_interest(frame)  # Use original colored frame
    cropped_time = (time.time() - start_time) * 1000

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 100, 255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Bitwise-AND mask and original image
    white_pixels = cv2.bitwise_and(frame, frame, mask=mask)

    #cv2.imshow("cropped", cropped_image)
    cv2.imshow("white_pixels", white_pixels)