import cv2
from prettytable import PrettyTable
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import json
import time

from piracer.cameras import Camera, MonochromeCamera
camera = MonochromeCamera()

#import for steering left and right
from driving import Functions_Driving
driving_instance = Functions_Driving()

#import line detection
from lane_detection import LaneDetection, main_lanes

class LaneDetector:
    def __init__(self):
        self.lane_detection = None

        self.total_time_sum = 0
        self.total_fps_sum = 0
        self.iterations = 1
        self.min_fps = 100

        #time data
        self.data = None

        self.angle = None

        frame = camera.read_image()
        self.result_frame = cv2.resize(frame, (1024, 768))

        self.straight = None

        self.lane_detection = LaneDetection(self.result_frame)

        self.start_time = None

        self.center_offset = None

        self.stream = None
        self.process = None
        self.motor = None
        self.debug = None
        self.placeholder = None

    def process_video(self):

        while True:

            frame = camera.read_image()
            frame = cv2.resize(frame, (1024, 768))

            self.detect_changes_in_json()

            #resize frame so it fits the camera reso
            frame = cv2.resize(frame, (1024, 768))

            #timer start for process time
            self.start_time = time.time()

            #lane detection
            self.center_offset, self.data, self.straight = main_lanes(frame, self.lane_detection, self.debug)

            #calculation for steering angle
            self.calculate_steering_angle()

            #process time
            total_time, fps, average_time, average_fps, min_fps = self.calculate_process_time()
            self.data = self.update_data(self.data, total_time, fps, average_time, average_fps, min_fps)
            
            #display result
            if self.process:
                self.print_table()
            self.iterations += 1

            if self.angle == 0:
                driving_instance.neutral_steering()
            elif self.angle < 0:
                driving_instance.left_steering(self.angle)
            else:
                driving_instance.left_steering(self.angle)
            if self.motor:
                driving_instance.frward_drive(0.15)

            if self.stream:
                #start stream here (TO-DO)
                pass


    def calculate_steering_angle(self):
        """
        Calculate steering angle from offset to middle
        :param offset: Offset from the middle (-250 to 250)
        :return: Steering angle in range [-1, 1]
        """
        if abs(self.center_offset) < 0:
            return 0.0

        if self.straight:
            normalized_offset = self.center_offset / 120
        else:
            normalized_offset = self.center_offset / 70

        self.angle = np.clip(normalized_offset, -1, 1)

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

    @staticmethod
    def update_data(data, total_time, fps, average_time, average_fps, min_fps):
        data.append(total_time)
        data.append(fps)
        data.append(average_time)
        data.append(average_fps)
        data.append(min_fps)
        return data

    def print_table(self):

        table = PrettyTable()
        table.field_names = ["White Time", "Region of Interest", "Transform", "Sliding Lane", "Car Position" , "Total Main Loop", "FPS", "Average Time", "Average FPS", "min FPS"]
        table.add_row(self.data)

        print(table)

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
    lane_detector.process_video()