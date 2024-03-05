
#import for steering left and right
#from driving import Functions_Driving
#driving_instance = Functions_Driving()

#import for angle
from trackline import Trackline
trackline_import = Trackline()

#import line detection
from hough_lanes import main_lanes

#import curve detection
from curve import detect_curve

#import for html website
from io import StringIO, BytesIO
import cv2
from logging import warning
from traceback import print_exc
from threading import Condition
from PIL import ImageFont, ImageDraw, Image
from http.server import BaseHTTPRequestHandler, HTTPServer
import copy
import time
from prettytable import PrettyTable
import numpy as np

#importing required OpenCV modules
from cv2 import COLOR_RGB2BGR, cvtColor

track = 0
centroids = 0
backup_centroids = 0
firstRun = True
call_counter = 0

def main():
    """
    Main function to process video frames, perform lane detection, and display results.
    """
    global track, centroids, backup_centroids, firstRun

    # Path to the video file
    video_path = 'no-off.avi'

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    total_time_sum = 0
    total_fps_sum = 0
    iterations = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1024, 768))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        start_time = time.time()

        if firstRun:
            # Get tracking points
            track_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            track, centroids, backup_centroids = getTrackline(track_frame)
            firstRun = False

        result, data, angle, steering_direction = main_lanes(frame, centroids, backup_centroids)

        #test for curve
        #detect_curve(frame)
                
        steering_angle = angle * 90

        steering_wheel_img = draw_steering_wheel(steering_angle, steering_direction)

        cv2.imshow('Steering Wheel', steering_wheel_img)        
        cv2.imshow('Video', result)

        total_time = (time.time() - start_time) * 1000
        fps = 1000/total_time
        total_time_sum += total_time
        iterations += 1
        total_fps_sum += fps
        average_time = total_time_sum / iterations
        average_fps = total_fps_sum / iterations

        data.append(total_time)
        data.append(fps)
        data.append(average_time)
        data.append(average_fps)

        #show battery percentage on display
        #driving_instance.battery_percent()

        #driving_instance.frward_drive(0.2)

        table_image = create_table_image(data)
        cv2.imshow('Process Time', table_image)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        

def getTrackline(frame):
    """
    Get the trackline from the given frame using predefined square regions.
    :param frame: The input frame
    :return: Tuple containing the trackline, centroids, and backup centroids
    """

    #define the coordinates and size of the squares
    square1 = (0, 0, 515, 770)  # (x, y, width, height)
    square2 = (550, 0, 514, 770) #rechts
    square3 = (0, 635, 1024, 180) #unten
    square4 = (0, 0, 1024, 570) #oben
    
    test = copy.deepcopy(frame)

    #draw the filled squares on the black image
    cv2.rectangle(test, (square1[0], square1[1]), (square1[0] + square1[2], square1[1] + square1[3]), color=(0, 0, 0), thickness=cv2.FILLED)
    cv2.rectangle(test, (square2[0], square2[1]), (square2[0] + square2[2], square2[1] + square2[3]), color=(0, 0, 0), thickness=cv2.FILLED)
    cv2.rectangle(test, (square3[0], square3[1]), (square3[0] + square3[2], square3[1] + square3[3]), color=(0, 0, 0), thickness=cv2.FILLED)
    cv2.rectangle(test, (square4[0], square4[1]), (square4[0] + square4[2], square4[1] + square4[3]), color=(0, 0, 0), thickness=cv2.FILLED)

    #apply main_detect function to get different video streams
    track, centroids, backup_centroids  = trackline_import.run(test)

    return track, centroids, backup_centroids

def create_table_image(data):
    """
    Creates an image of a table displaying the given data.
    :param data: List containing the data to be displayed in the table
    :return: Image of the table in OpenCV format
    """

    for i in range(len(data)):
        if isinstance(data[i], float):
            data[i] = round(data[i], 2)
            if i != 7 and i != 9:
                data[i] = f'{data[i]} ms'

    table = PrettyTable()
    table.field_names = ["Timestamp", "Canny Image", "Region of Interest", "Hough Lines", "Line Image", "Total Image Time", "Total Main Loop", "FPS", "Average Time", "Average FPS"]
    table.add_row(data)

    min_widths = {"Timestamp": 10, "Canny Image": 10, "Region of Interest": 10, "Hough Lines": 10, "Line Image": 10, "Total Image Time": 10, "Total Main Loop": 10, "FPS": 10, "Average Time" : 10, "Average FPS" : 10}
    for field in table.field_names:
        table._min_width[field] = min_widths.get(field, 0)

    table_str = str(table)  # Convert the PrettyTable to a string
    font = ImageFont.load_default()  # Load default font

    # Create a new blank image with white background and increased width
    img = Image.new('RGB', (950, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    # Write the table content onto the image
    d.text((10, 10), table_str, fill=(0, 0, 0), font=font)

    # Convert the PIL image to OpenCV format
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return cv_img


def draw_steering_wheel(angle, steering_direction):
    """
    Draws a steering wheel image with the specified angle.
    :param angle: Angle to rotate the steering wheel (in degrees)
    :return: Image of the rotated steering wheel with angle annotation
    """
    # Load the steering wheel image
    steering_wheel_img = cv2.imread('steering-wheel.png', cv2.IMREAD_UNCHANGED)

    # Resize the steering wheel image to half its original size
    steering_wheel_img = cv2.resize(steering_wheel_img, (0, 0), fx=0.5, fy=0.5)

    height, width = steering_wheel_img.shape[:2]

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

    # Rotate the steering wheel image
    rotated_steering_wheel = cv2.warpAffine(steering_wheel_img, rotation_matrix, (width, height))

    # Create a white background image
    white_background = np.ones_like(steering_wheel_img) * 255

    # Blend the steering wheel with the white background
    blended_image = cv2.addWeighted(white_background, 0.0, rotated_steering_wheel, 1.0, 0)

    # Add text annotations
    cv2.putText(blended_image, f"Angle: {angle}", (20, blended_image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(blended_image, f"Steering Direction: {steering_direction}", (20, blended_image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return blended_image

if __name__ == "__main__":
    main()