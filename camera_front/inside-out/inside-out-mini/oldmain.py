
#import for steering left and right
#from driving import Functions_Driving
#driving_instance = Functions_Driving()

#import for angle
from trackline import Trackline
trackline_import = Trackline()

#import line detection
from lane_detection import LaneDetection, main_lanes

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

from piracer.cameras import Camera, MonochromeCamera
camera = MonochromeCamera()

#import for steering left and right
from driving import Functions_Driving
driving_instance = Functions_Driving()

call_counter = 0

def main():

    lane_detection = LaneDetection()

    total_time_sum = 0
    total_fps_sum = 0
    iterations = 0
    min_fps = 100

    # Loop through the video frames
    while True:
        """
        Loop continuously while the video capture is open. 
        Read frames from the video, resize them, and perform main lane detection.
        Calculate steering angle based on lane center offset and draw the steering wheel image accordingly.
        Display the steering wheel and process time (including FPS and average time) on separate windows.
        Press 'q' to exit the loop.
        """
        # Read a frame from the video
        frame = camera.read_image()
        frame = cv2.resize(frame, (1024, 768))
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        #timer start for process time
        start_time = time.time()

        #lane detection
        center_offset, data, straight = main_lanes(frame, lane_detection)
        
        #calculation for steering angle
        angle = calculate_steering_angle(center_offset, straight)

        print(f'angle: {angle}')
        #print(angle)
        #steering_angle = angle * 90

        #steering_wheel_img = draw_steering_wheel(steering_angle)
        #cv2.imshow('Steering Wheel', steering_wheel_img)        

        #calculations for process time
        total_time = (time.time() - start_time) * 1000
        fps = 1000/total_time
        if fps < min_fps and iterations > 50:
            min_fps = fps
        total_time_sum += total_time
        iterations += 1
        total_fps_sum += fps
        average_time = total_time_sum / iterations
        average_fps = total_fps_sum / iterations

        #append process times to data
        data.append(total_time)
        data.append(fps)
        data.append(average_time)
        data.append(average_fps)
        data.append(min_fps)

        #show battery percentage on display
        #driving_instance.battery_percent()
        #print(angle)

        if angle == 0:
            driving_instance.neutral_steering()
        elif angle < 0:
            driving_instance.left_steering(angle)
        else:
            driving_instance.left_steering(angle)
        driving_instance.frward_drive(0.15)

        print_table(data)
        #table_image = create_table_image(data)
        #cv2.imshow('Process Time', table_image)
        

def calculate_steering_angle(offset, straight):
    """
    Calculate steering angle from offset to middle
    
    :param offset: Offset from the middle (-50 to 50)
    :return: Steering angle in range [0, 1]
    """
    # Clamp offset to range [-50, 50]
    offset = max(-250, min(offset, 250))
    
    # Check if offset is smaller than 10 percent
    if abs(offset) < 0:
        return 0.0
    
    # Normalize offset to range [-1, 1]
    if straight:
        normalized_offset = offset / 60
        angle = normalized_offset
    else:
        normalized_offset = offset / 250
        angle = normalized_offset

    angle = angle * -1
    #max
    #if angle > 1:
        #angle = 1
    #elif angle < -1:
        #angle = -1
    
    return angle

def print_table(data):

    table = PrettyTable()
    table.field_names = ["Timestamp", "White Time", "Region of Interest", "Transform", "Sliding Lane", "Car Position" , "Total Main Loop", "FPS", "Average Time", "Average FPS", "min FPS"]
    table.add_row(data)

    print(table)


if __name__ == "__main__":
    main()