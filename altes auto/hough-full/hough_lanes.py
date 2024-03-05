import cv2
import numpy as np
import math
import time

#create an instance of Functions_drivin
from driving import Functions_Driving
driving_instance = Functions_Driving()

#create an instance of SteeringController
from steering import SteeringController
steering_controller = SteeringController(driving_instance)

backup_left_line = np.array([0, 0, 0, 0])
backup_right_line = np.array([0, 0, 0, 0])
coords_left = [0, 0, 0, 0]
coords_right = [1000,500,1000,500]

def make_coordinates(image, line_parameters, position):
    global coords_left, coords_right

    #generate a single averaged line for the left and right lane by obtaining the average coordinates from both the left and right lines.
    try:
        #attempt to unpack line_parameters
        slope, intercept = line_parameters
        y1 = image.shape[0]        
        y2 = int(y1 * (1/2))  # length of displayed line
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        if position == "left":
            coords_left = [x1, y1, x2, y2]
        else:
            coords_right = [x1, y1, x2, y2]

        return np.array([x1, y1, x2, y2])

    except TypeError as e:
        if position == "left":
            coords_left = backup_left_line
            print("backup")
            return backup_left_line
        else:
            coords_right = backup_right_line
            print("backup")
            return backup_right_line

def average_slope_intercept(image, lines):
    global backup_left_line, backup_right_line

    left_fit = []
    right_fit = []

    #sort the lane coordinates, generated using HoughLines, based on their slopes to differentiate between left and right lanes.
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    #generate a single averaged line for the left and right lane by obtaining the average coordinates from both the left and right lines.
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_average, "left")
    right_line = make_coordinates(image, right_fit_average, "right")

    #save the lane coordinates as a backup so that in the absence of a detected line, the backup can be utilized temporarily until a line is detected again.
    backup_left_line = left_line
    backup_right_line = right_line

    print(left_line)
    print(right_line)

    return np.array([left_line, right_line])


def canny(image):
    #get canny image (remove blur and apply scale)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines, centroids, backup_centroids, distance = 70):
    line_image = np.zeros_like(image)

    if lines is not None:

        #draw left and right lane (blue)
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)

        #coords of the left blue lane
        x_leftTop = coords_left[0]
        x_leftBottom = coords_left[2]
        y_leftTop = coords_left[1]
        y_leftBottom = coords_left[3]

        #coords of the right blue line
        x_rightTop = coords_right[0]
        x_rightBottom = coords_right[2]
        y_rightTop = coords_right[1]
        y_rightBottom = coords_right[3]

        #calculate midpoints xu, yu, xo, yo
        xu = int(x_leftTop + (x_rightTop - x_leftTop) / 2)
        yu = int(y_rightTop)
        xo = int(x_leftBottom + (x_rightBottom - x_leftBottom) / 2)
        yo = int(y_rightBottom)

        #draw a line connecting midpoints
        cv2.line(line_image, (xu, yu), (xo, yo), (255, 0, 0), 10)

        #calculate the angle of the line
        if len(centroids) >= 2:
            angle = math.atan2(centroids[1][1] - centroids[0][1], centroids[0][0] - centroids[1][0])
        else:
            angle = math.atan2(backup_centroids[1][1] - backup_centroids[0][1], backup_centroids[0][0] - backup_centroids[1][0])

        #calculate the coordinates of a point at a distance from the midpoint
        px1 = int(centroids[0][0] + distance * math.cos(angle))
        py1 = int(centroids[0][1] + distance * math.sin(angle))

        px2 = int(centroids[1][0] - distance * math.cos(angle))
        py2 = int(centroids[1][1] - distance * math.sin(angle))

        #draw line of the direction of the car
        cv2.line(line_image, (px2, py2), (px1, py1), (0, 255, 0), 10)

        #line tuple of the left and right lane (blue)
        line_right = (x_rightBottom, y_rightBottom, x_rightTop, y_rightTop)
        line_left = (x_leftBottom, y_leftBottom, x_leftTop, y_leftTop)

        #calculate steering based on midpoint of car, left and right line
        steering_controller.distance_line_to_point(px2, py2, line_left, line_right)

    return line_image


def region_of_interest(image):
    #height of the stream
    height = image.shape[0]
    width = image.shape[1]

    print(f'{width}x{height}')

    #define region 
    polygons = np.array([
        [(0, height-50), (1100, height-50), (500, 250)]
        ])
    #apply region of interest (polygon is punched out)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and( image, mask)
    return masked_image



def VideoCapture(frame, centroids, backup_centroids):
    start_time = time.time()
    
    canny_image = canny(frame)
    canny_time = (time.time() - start_time) * 1000
    print("Canny Edge Detection Time:", canny_time, "ms")
    
    # Define the vertices of the triangle
    pts = np.array([[100, 800], [900, 800], [500, 400]], dtype=np.int32)
    pts2 = np.array([[100, 350], [900, 350], [500, 250]], dtype=np.int32)

    # Reshape the array into a 3D array with one row
    pts = pts.reshape((-1, 1, 2))
    pts2 = pts2.reshape((-1, 1, 2))

    # Draw the filled triangle on the black image
    cv2.fillPoly(canny_image, [pts, pts2], color=(0, 0, 0))

    cropped_image = region_of_interest(canny_image)
    cropped_time = (time.time() - start_time) * 1000
    print("Region of Interest Time:", cropped_time - canny_time, "ms")
    
    # detects lines with houghmesh
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=10)
    hough_time = (time.time() - start_time) * 1000
    print("Hough Transformation Time:", hough_time - cropped_time, "ms")
    
    if lines is not None:
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines, centroids, backup_centroids)
    else:
        # If no lines detected, use the original frame.
        line_image = np.copy(frame)
    
    line_time = (time.time() - start_time) * 1000 
    print("Line Display Time:", line_time - hough_time, "ms")
        
    # blend lanes over original image
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    blend_time = (time.time() - start_time) * 1000  #combo time
    print("Image Blending Time:", blend_time - line_time, "ms")
    
    total_time = (time.time() - start_time) * 1000  #Total time
    print("Total Function Time:", total_time, "ms")
    
    return canny_image, cropped_image, line_image, combo_image


def main_lanes(frame, centroids, backup_centroids):

    canny, field_of_interest, detected_lanes, result = VideoCapture(frame, centroids, backup_centroids)

    #return generated images
    return canny, field_of_interest, detected_lanes, result