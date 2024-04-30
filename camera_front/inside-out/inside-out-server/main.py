import cv2
from prettytable import PrettyTable
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import json
import time
import os
from calib_cam import calib
from sender import sender_frame, putQueue
import threading

#import for steering left and right
from driving import Functions_Driving
driving_instance = Functions_Driving()

#import line detection
from lane_detection import LaneDetection, main_lanes

#set camera filter and rotation at start
device_path='/dev/video0'
os.system(f'v4l2-ctl -d {device_path} --set-ctrl=rotate={180}')
os.system(f'v4l2-ctl -d {device_path} --set-ctrl=color_effects=1') # Run 'v4l2-ctl -L' for explanations

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

        self.cap = cv2.VideoCapture(0)
        #frame = camera.read_image()
        #self.result_frame = cv2.resize(frame, (1024, 768))

        self.curve = None
        self.speed_curve = None

        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (1024, 768))
        #frame = cv2.rotate(frame, cv2.ROTATE_180)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.lane_detection = LaneDetection(frame)

        self.start_time = None

        self.center_offset = None

        self.stream = None
        self.process = None
        self.motor = None
        self.debug = None
        self.placeholder = None

        self.steering_offset = 0.21

        frame_width = 1024
        frame_height = 768
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.last_curve_tester = None
        self.curve_tester = None
        self.tester = False

        #self.fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for saving video (XVID for .avi)
        #self.out = cv2.VideoWriter('output_video.avi', self.fourcc, fps, (frame_width, frame_height))

    def process_video(self):

        while True:

            #timer start for process time
            self.start_time = time.time()

            self.detect_changes_in_json()

            #lane detection
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (1024, 768))

            calib_time = time.time()
            #calibrate cam image
            #frame = calib(frame)
            calibration_time = (time.time() - calib_time) * 1000

            self.center_offset, self.data, self.curve, self.speed_curve = main_lanes(frame, self.lane_detection, self.debug, self.placeholder)

            #calculation for steering angle
            self.calculate_steering_angle()

            #process time
            total_time, fps, average_time, average_fps, min_fps = self.calculate_process_time()
            self.data = self.update_data(self.data, total_time, fps, average_time, average_fps, min_fps, calibration_time)
            
            #display result
            if self.process:
                self.print_table()
            self.iterations += 1

            if self.angle == 0:
                driving_instance.left_steering(-self.steering_offset)
            elif self.angle < 0:
                driving_instance.left_steering(self.angle - self.steering_offset)
            else:
                driving_instance.left_steering(self.angle - self.steering_offset)
            if self.motor:
                if self.tester == False:
                    self.last_curve_tester = self.speed_curve

                if self.speed_curve == False and self.last_curve_tester == True:
                    self.tester = True
                    if self.curve_tester == None:
                        self.curve_tester = 1
                    elif self.curve_tester < 20:
                        self.curve_tester += 1
                        self.speed_curve = True
                    else:
                        self.curve_tester = None
                        self.speed_curve = False
                        self.tester = False

                if self.speed_curve or self.curve:
                    #print("--------------------------------------------------------------------------")
                    driving_instance.frward_drive(0.13) #0.4
                else:
                    #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    driving_instance.frward_drive(0.13) #0.56
            else:
                driving_instance.frward_drive(0)

            putQueue(frame)

            #self.out.write(frame)
            #self.save_number_to_file(self.angle)

    def save_number_to_file(self,number):
        filename = 'angle.txt'
        try:
            with open(filename, 'a') as file:  # Use 'a' mode to append to the file
                file.write(str(number) + '\n')
            print(f"Number {number} successfully saved to {filename}")
        except Exception as e:
            print(f"Error occurred while saving number {number}: {e}")

    def calculate_steering_angle(self):
        """
        Calculate steering angle from offset to middle
        :param offset: Offset from the middle (-250 to 250)
        :return: Steering angle in range [-1, 1]
        """

        if self.curve:
            normalized_offset = self.center_offset / 350
            #if normalized_offset > 0:
            #    normalized_offset = 0
        else:
            normalized_offset = (self.center_offset) / 800

        #normalized_offset *= -1

        self.angle = np.clip(normalized_offset, -0.9, 0.9)

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
    def update_data(data, total_time, fps, average_time, average_fps, min_fps, calibration_time):
        data.append(calibration_time)
        data.append(total_time)
        data.append(fps)
        data.append(average_time)
        data.append(average_fps)
        data.append(min_fps)
        return data

    def print_table(self):

        self.data = [round(num, 3) if isinstance(num, float) else num for num in self.data]

        table = PrettyTable()
        table.field_names = ["Rows selection", "Transform", "Sliding Lane", "Car Position" , "Calibration" , "Total Main Loop", "FPS", "Average Time", "Average FPS", "min FPS"]
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
    #start the backround-thread for the frame
    main_thread = threading.Thread(target=lane_detector.process_video)
    stream_thread = threading.Thread(target=sender_frame)
    main_thread.start()
    stream_thread.start()