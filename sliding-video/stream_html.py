
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
import time
from prettytable import PrettyTable
import numpy as np

#importing required OpenCV modules
from cv2 import COLOR_RGB2BGR, cvtColor

call_counter = 0

def main():

    # Path to the video file
    video_path = '/home/marvin/Desktop/sliding-video/videos/no-off.avi'

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not opeqn video.")
        exit()

    total_time_sum = 0
    total_fps_sum = 0
    iterations = 0
    min_fps = 100

    # Loop through the video frames
    while cap.isOpened():
        """
        Loop continuously while the video capture is open. 
        Read frames from the video, resize them, and perform main lane detection.
        Calculate steering angle based on lane center offset and draw the steering wheel image accordingly.
        Display the steering wheel and process time (including FPS and average time) on separate windows.
        Press 'q' to exit the loop.
        """
        # Read a frame from the video
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1024, 768))
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        #timer start for process time
        start_time = time.time()

        #lane detection
        center_offset, data, straight = main_lanes(frame)
        
        #calculation for steering angle
        angle = calculate_steering_angle(center_offset, straight)
        #print(angle)
        steering_angle = angle * 90

        steering_wheel_img = draw_steering_wheel(steering_angle)
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

        #driving_instance.frward_drive(0.2)

        table_image = create_table_image(data)
        cv2.imshow('Process Time', table_image)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        

def calculate_steering_angle(offset, straight):
    """
    Calculate steering angle from offset to middle
    
    :param offset: Offset from the middle (-50 to 50)
    :return: Steering angle in range [0, 1]
    """
    # Clamp offset to range [-50, 50]
    offset = max(-15, min(offset, 15))
    
    # Check if offset is smaller than 10 percent
    if abs(offset) < 2:
        return 0.0
    
    # Normalize offset to range [-1, 1]
    if straight:
        normalized_offset = offset / 60
        angle = normalized_offset
    else:
        normalized_offset = offset / 30
        angle = normalized_offset

    #max
    if angle > 1:
        angle = 1
    elif angle < -1:
        angle = 1
    
    return angle

def create_table_image(data):
    """
    Creates an image of a table displaying the given data.
    :param data: List containing the data to be displayed in the table
    :return: Image of the table in OpenCV format
    """

    for i in range(len(data)):
        if isinstance(data[i], float):
            data[i] = round(data[i], 2)
            if i != 10 and i != 12 and i != 13:
                data[i] = f'{data[i]} ms'

    table = PrettyTable()
    table.field_names = ["Timestamp", "White Time", "No Car", "Region of Interest", "Transform", "Histogram", "Sliding Lane", "Overlay Lines" , "Car Position" , "Total Main Loop", "FPS", "Average Time", "Average FPS", "min FPS"]
    table.add_row(data)

    min_widths = {"Timestamp" :10,"White Time" :10, "No Car" :10, "Region of Interest" :10, "Transform" :10, "Histogram" :10, "Sliding Lane" :10, "Overlay Lines" :10, "Car Position" :10, "Total Main Loop" :10, "FPS" :10, "Average Time" :10, "Average FPS" :10, "min FPS" :10}
    for field in table.field_names:
        table._min_width[field] = min_widths.get(field, 0)

    table_str = str(table)  # Convert the PrettyTable to a string
    font = ImageFont.load_default()  # Load default font

    # Create a new blank image with white background and increased width
    img = Image.new('RGB', (1250, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    # Write the table content onto the image
    d.text((10, 10), table_str, fill=(0, 0, 0), font=font)

    # Convert the PIL image to OpenCV format
    table_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return table_image


def draw_steering_wheel(angle):
    """
    Draws a steering wheel image with the specified angle.
    :param angle: Angle to rotate the steering wheel (in degrees)
    :return: Image of the rotated steering wheel with angle annotation
    """
    # Load the steering wheel image
    steering_wheel_img = cv2.imread('test_images/steering-wheel.png', cv2.IMREAD_UNCHANGED)

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
    #cv2.putText(blended_image, f"Steering Direction: {steering_direction}", (20, blended_image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return blended_image

if __name__ == "__main__":
    main()