
#import for steering left and right
#from driving import Functions_Driving
#driving_instance = Functions_Driving()

#import for angle
from trackline import Trackline
trackline_import = Trackline()

#import line detection
from hough_lanes import main_lanes

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


# Path to the video file
VIDEO_FILE = "videos/no-off.mp4"

def main(firstRun = False):
    global track, centroids, backup_centroids

    video_capture = cv2.VideoCapture(VIDEO_FILE)

    # Check if the video file is opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return

    # Read and display frames from the video file
    while True:
        # Read a frame from the video file
        ret, frame = video_capture.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error: Failed to read frame.")
            break

    start_time = time.time()

    if firstRun:
        #get tracking points
        track, centroids, backup_centroids = getTrackline(frame)

    #apply main_detect function to get different video streams
    result, data, angle, steering_direction = main_lanes(frame, centroids, backup_centroids)

    total_time = (time.time() - start_time) * 1000  #Total time

    fps = 1000/total_time
    data.append(total_time)
    data.append(fps)

    #show battery percentage on display
    #driving_instance.battery_percent()

    #driving_instance.frward_drive(0.2)

    table_image = create_table_image(data)  # Create table image
    cv2.imshow('Table', table_image)  # Show table image

    steering_angle = angle * 90

    steering_wheel_img = draw_steering_wheel(steering_angle, steering_direction)
    cv2.imshow('Steering Wheel', steering_wheel_img)        

    cv2.imshow('Video', result)
        

def getTrackline(frame):

    #define the coordinates and size of the squares
    square1 = (0, 0, 500, 770)  # (x, y, width, height)
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
    table = PrettyTable()
    table.field_names = ["Timestamp", "Canny Image", "Region of Interest", "Hough Lines", "Line Image", "Total Image Time", "Total Main Loop", "FPS"]
    table.add_row(data)

    table_str = str(table)  # Convert the PrettyTable to a string
    font = ImageFont.load_default()  # Load default font

    # Create a new blank image with white background
    img = Image.new('RGB', (800, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    # Write the table content onto the image
    d.text((10, 10), table_str, fill=(0, 0, 0), font=font)

    # Convert the PIL image to OpenCV format
    cv_img = cvtColor(np.array(img), COLOR_RGB2BGR)

    return cv_img

def draw_steering_wheel(angle, steering_direction):
    steering_wheel_img = cv2.imread('steering-wheel.png')

    height, width = steering_wheel_img.shape[:2]

    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)

    rotated_steering_wheel = cv2.warpAffine(steering_wheel_img, rotation_matrix, (width, height))

    cv2.putText(rotated_steering_wheel, f"Angle: {angle}", (20, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(rotated_steering_wheel, f"Steering Direction: {steering_direction}", (20, height + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return rotated_steering_wheel


if __name__ == "__main__":
    main(True)
    while True:
        main()

    cv2.destroyAllWindows()
