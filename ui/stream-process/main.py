import cv2
from prettytable import PrettyTable
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import json
import time
import threading
from display import displayImages, putQueue, FourFrameWindow

#import line detection
from lane_detection import LaneDetection, main_lanes

from calibra import calib

class LaneDetector:
    def __init__(self):

        video_path = '/home/marvin/Desktop/stream-process/output_video.avi'
        self.cap = cv2.VideoCapture(video_path) 

        self.lane_detection = None

        self.total_time_sum = 0
        self.total_fps_sum = 0
        self.iterations = 1
        self.min_fps = 100

        #time data
        self.data = None

        self.angle = None

        self.curve = None

        ret, frame = self.cap.read()
        self.frame = cv2.resize(frame, (1024, 768))

        self.lane_detection = LaneDetection(self.frame)

        self.start_time = None

        self.center_offset = None

        self.stream = None
        self.process = None
        self.motor = None
        self.debug = None
        self.placeholder = None

        self.cropped_image = None

        self.steering_offset = 0.21

        self.four_frame_window = FourFrameWindow()

    def process_video(self):

        while True:

            #timer start for process time
            self.start_time = time.time()

            self.detect_changes_in_json()

            _, frame = self.cap.read()
            frame = cv2.resize(frame, (1024, 768))
            self.frame = calib(frame)

            self.center_offset, self.data, self.curve, self.speed_curve, self.cropped_image, data_car = main_lanes(self.frame, self.lane_detection, self.debug, self.placeholder)

            #calculation for steering angle
            self.calculate_steering_angle()

            #process time
            total_time, fps, average_time, average_fps, min_fps = self.calculate_process_time()
            self.data = self.update_data(self.data, total_time, fps, average_time, average_fps, min_fps)

            self.update_queue()
            displayImages(self.four_frame_window, data_car)
            self.iterations += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def calculate_steering_angle(self):
        """
        Calculate steering angle from offset to middle
        :param offset: Offset from the middle (-250 to 250)
        :return: Steering angle in range [-1, 1]
        """
        if self.curve:
            normalized_offset = self.center_offset / 100
            #if normalized_offset > 0:
            #    normalized_offset = 0
        else:
            normalized_offset = self.center_offset / 600

        #normalized_offset *= -1

        self.angle = np.clip(normalized_offset, -0.9, 0.9)

        self.angle = self.angle - self.steering_offset

    def calculate_process_time(self):
        total_time = (time.time() - self.start_time) * 1000
        fps = 1000 / total_time
        self.total_fps_sum += fps
        if fps < self.min_fps and self.iterations > 50:
            self.min_fps = fps
        self.total_time_sum += total_time
        average_time = self.total_time_sum / self.iterations
        average_fps = self.total_fps_sum / self.iterations
        return total_time, fps, average_time, average_fps, self.min_fps

    def update_queue(self):
        table_image = self.create_table_image()
        steering_wheel = self.draw_steering_wheel()
        orig_frame = self.frame
        detected_lanes = self.cropped_image

        putQueue([table_image, steering_wheel, orig_frame, detected_lanes])

    @staticmethod
    def update_data(data, total_time, fps, average_time, average_fps, min_fps):
        data.append(total_time)
        data.append(fps)
        data.append(average_time)
        data.append(average_fps)
        data.append(min_fps)
        return data


    def create_table_image(self):
        """
        Creates an image of a table displaying the given data.
        :param data: List containing the data to be displayed in the table
        :return: Image of the table in OpenCV format
        """

        for i in range(len(self.data)):
            if isinstance(self.data[i], float):
                self.data[i] = round(self.data[i], 2)
                if i != 7 and i != 9:
                    self.data[i] = f'{self.data[i]} ms'

        table = PrettyTable()
        table.field_names = ["Rows selection", "Transform", "Sliding Lane", "Car Position" , "Total Main Loop", "FPS", "Average Time", "Average FPS", "min FPS"]

        table.add_row(self.data)

        min_widths = {"Rows selection" : 10, "Transform" : 10, "Sliding Lane" : 10, "Car Position" :10, "Total Main Loop" : 10, "FPS" : 10, "Average Time" : 10, "Average FPS" : 10, "min FPS" : 10}

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


    def draw_steering_wheel(self):
        """
        Draws a steering wheel image with the specified angle.
        :param angle: Angle to rotate the steering wheel (in degrees)
        :return: Image of the rotated steering wheel with angle annotation
        """
        # Load the steering wheel image
        steering_wheel_img = cv2.imread('wheel.jpg', cv2.IMREAD_UNCHANGED)

        # Resize the steering wheel image to half its original size
        steering_wheel_img = cv2.resize(steering_wheel_img, (0, 0), fx=1, fy=1)

        height, width = steering_wheel_img.shape[:2]

        # Compute the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), self.angle*90, 1)

        # Rotate the steering wheel image
        rotated_steering_wheel = cv2.warpAffine(steering_wheel_img, rotation_matrix, (width, height))

        # Create a white background image
        white_background = np.ones_like(steering_wheel_img) * 255

        # Blend the steering wheel with the white background
        blended_image = cv2.addWeighted(white_background, 0.0, rotated_steering_wheel, 1.0, 0)

        # Add text annotations
        cv2.putText(blended_image, f"Angle: {self.angle}", (20, blended_image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv2.putText(blended_image, f"Steering Direction: {steering_direction}", (20, blended_image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return blended_image


    def read_json(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        return data

    def detect_changes_in_json(self):
        current_data = self.read_json('data.json')
        self.stream = current_data.get('stream', False)
        self.process = current_data.get('process', False)
        self.motor = current_data.get('motor', False)
        self.debug = current_data.get('debug', False)
        self.placeholder = current_data.get('placeholder', False)

if __name__ == "__main__":
    lane_detector = LaneDetector()
    #start the backround-thread for the frame
    lane_detector.process_video()