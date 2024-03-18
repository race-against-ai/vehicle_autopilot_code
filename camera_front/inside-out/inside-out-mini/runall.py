from lane_detection import LaneDetection, main_lanes
from vector import Vector
from steer import PIDController, steer
from timer import Timer
#from stream import Streamingmodule

import json
import cv2
import time

from piracer.cameras import Camera, MonochromeCamera
camera = MonochromeCamera()


class Auto:

    def __init__(self):

        frame = camera.read_image()
        self.result_frame = cv2.resize(frame, (1024, 768))

        self.lane_detection = LaneDetection(self.result_frame)

        self.vector = Vector()

        self.time = Timer()

        self.stream = None
        self.process = None
        self.motor = None
        self.debug = None
        self.placeholder = None

        self.time_data = None

        self.curve = None

        self.left_and_right_points = None


    def process_video(self):

        while True:

            start_time = time.time()

            #this will enable feautures of the car
            self.detect_changes_in_json()

            #provides following info: car offset, process times, data of left and right line
            self.line_informations()

            #provides following info: curve
            self.curve = self.detect_curve()

            #provides following info: steering angle
            self.steering()
            
            #provides following info: process time
            if self.process:
                self.time.print_table(self.time_data, start_time)

            #provides following info: stream of camera
            if self.stream:
                pass

    def line_informations(self):

        frame = camera.read_image()
        frame = cv2.resize(frame, (1024, 768))

        #lane detection gets process time
        self.center_offset, self.time_data, self.left_and_right_points = main_lanes(frame, self.lane_detection, self.debug)


    def detect_curve(self, printToTerminal=False):

        left_counts, right_counts = self.left_and_right_points

        if left_counts is not None:

            print(left_counts)

            left_some_512 = any(count == 512 for count in left_counts)
            if left_some_512:
                left = next((val for val in reversed(left_counts) if val != 512), None)
                point_to_check = (left,296)
            else:
                point_to_check = (left_counts[-1],296)

            point1 = (left_counts[0],767)
            point2 = (left_counts[1],660)
            distance = self.vector.calculate_distance(point1,point2,point_to_check, debug=False)

            if distance < 50:
                if printToTerminal:
                    print(f'Offset einer Linie für Kurve {distance}')
                    print('Gerade')
                return False
            else:
                if printToTerminal:
                    print(f'Offset einer Linie für Kurven {distance}')
                    print('Kurve')
                return True

    def steering(self):
        steer(self.motor, self.curve, self.center_offset)

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
    lane_detector = Auto()
    lane_detector.process_video()