import cv2
from prettytable import PrettyTable
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import json
import time

#import line detection
from lane_detection import LaneDetection, main_lanes

class LaneDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.lane_detection = None

        self.total_time_sum = 0
        self.total_fps_sum = 0
        self.iterations = 1
        self.min_fps = 100

        self.high_white = 0
        self.high_roi = 0
        self.high_transform = 0
        self.high_find = 0
        self.high_dashed = 0

        self.total_white_time = 0
        self.total_roi_time = 0
        self.total_transform_time = 0
        self.total_find_time = 0
        self.total_dashed_time = 0

        #time data
        self.data = None

        self.angle = None

        _, self.result_frame = self.cap.read()
        self.result_frame = cv2.resize(self.result_frame, (1024, 768))

        self.straight = None

        self.lane_detection = LaneDetection(self.result_frame)

        self.start_time = None

        self.center_offset = None

        self.stream = None
        self.process = None
        self.motor = None
        self.debug = None
        self.placeholder = None

        self.second_row_data = []

        self.third_row_data = []

    def process_video(self):

        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

        while self.cap.isOpened():
            # Read a frame from the video
            ret, frame = self.cap.read()
            if not ret:
                break

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
                self.display_results()
            self.iterations += 1

            #if self.angle == 0:
                #driving_instance.neutral_steering()
            #elif self.angle < 0:
                #driving_instance.left_steering(self.angle)
            #else:
                #driving_instance.left_steering(self.angle)
            if self.motor:
                #driving_instance.frward_drive(0.15)
                pass

            if self.stream:
                #start stream here (TO-DO)
                pass

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    def calculate_steering_angle(self):
        """
        Calculate steering angle from offset to middle
        :param offset: Offset from the middle (-250 to 250)
        :return: Steering angle in range [-1, 1]
        """
        if abs(self.center_offset) < 0:
            return 0.0

        if self.straight:
            normalized_offset = self.center_offset / 150
        else:
            normalized_offset = self.center_offset / 100

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
        self.total_white_time += self.data[1]
        self.total_roi_time += self.data[2]
        self.total_transform_time += self.data[3]
        self.total_find_time += self.data[4]
        self.total_dashed_time += self.data[5]

        average_white = self.total_white_time / self.iterations
        average_roi = self.total_roi_time / self.iterations
        average_transform = self.total_transform_time / self.iterations
        average_find = self.total_find_time / self.iterations
        average_dashed = self.total_dashed_time / self.iterations

        if self.data[1] > self.high_white:
            self.high_white = self.data[1]
        if self.data[2] > self.high_roi:
            self.high_roi = self.data[2]
        if self.data[3] > self.high_transform:
            if self.iterations > 100:
                self.high_transform = self.data[3]
        if self.data[4] > self.high_find:
            self.high_find = self.data[4]
        if self.data[5] > self.high_dashed:
            self.high_dashed = self.data[5]

        self.second_row_data = 0, average_white, average_roi, average_transform, average_find, average_dashed, 0,0,0,0,0
        self.third_row_data = 0, self.high_white,  self.high_roi,  self.high_transform, self.high_find, self.high_dashed, 0,0,0,0,0
        return total_time, fps, average_time, average_fps, self.min_fps

    @staticmethod
    def update_data(data, total_time, fps, average_time, average_fps, min_fps):
        data.append(total_time)
        data.append(fps)
        data.append(average_time)
        data.append(average_fps)
        data.append(min_fps)
        return data

    def display_results(self):
        table_image = self.create_table_image()
        cv2.imshow('Process Time', table_image)
        print('Angle',self.angle)


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

        self.second_row_data = list(self.second_row_data)
        for i in range(len(self.second_row_data)):
            if isinstance(self.second_row_data[i], float):
                self.second_row_data[i] = round(self.second_row_data[i], 2)

        self.third_row_data = list(self.third_row_data)
        for i in range(len(self.third_row_data)):
            if isinstance(self.third_row_data[i], float):
                self.third_row_data[i] = round(self.third_row_data[i], 2)

        table = PrettyTable()
        table.field_names = ["Timestamp", "White Time", "Region of Interest", "Transform", "Inside-Out", "Dashed Calculation", "Total Main Loop", "FPS", "Average Time", "Average FPS", "min FPS"]

        table.add_row(self.data)
        table.add_row(self.second_row_data)
        table.add_row(self.third_row_data)

        min_widths = {"Timestamp": 10, "White Time": 10, "Region of Interest": 10, "Transform": 10, "Inside-Out": 10, "Dashed Calculation": 10, "Total Main Loop": 10, "FPS": 10, "Average Time": 10, "Average FPS": 10, "min FPS": 10}

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
        steering_wheel_img = cv2.imread('test_images/steering-wheel.png', cv2.IMREAD_UNCHANGED)

        # Resize the steering wheel image to half its original size
        steering_wheel_img = cv2.resize(steering_wheel_img, (0, 0), fx=0.5, fy=0.5)

        height, width = steering_wheel_img.shape[:2]

        # Compute the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), self.angle, 1)

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
    video_path = '/home/marvin/Desktop/sliding-video/videos/no-off.avi'
    lane_detector = LaneDetector(video_path)
    lane_detector.process_video()