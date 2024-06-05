import cv2
from prettytable import PrettyTable
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import json
import time
import threading
from display import displayImages, FourFrameWindow

#import line detection
from lane_detection import LaneDetection, main_lanes

from calibra import calib

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

        self.curve = None

        self.frame = np.zeros((720, 960, 3), dtype=np.uint8)
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

        # Start the camera
        stream_url = "http://192.168.30.123:8443/canny_html.py"
        #stream_url = 'localstream\schieben.avi'
        camera = cv2.VideoCapture(stream_url)

        # Start the cleaning thread
        self.cam_cleaner = CameraBufferCleanerThread(camera)

    def process_video(self):

        while True:

            #timer start for process time
            self.start_time = time.time()

            self.detect_changes_in_json()

            if self.cam_cleaner.last_frame is not None:
                self.frame = self.cam_cleaner.last_frame

            print(f'test {self.frame.shape}')

            self.center_offset, self.data, self.curve, self.speed_curve, self.cropped_image, data_car = main_lanes(self.frame, self.lane_detection, self.debug)

            #calculation for steering angle

            self.calculate_steering_angle()

            #process time
            #total_time, fps, average_time, average_fps, min_fps = self.calculate_process_time()
            #self.data = self.update_data(self.data, total_time, fps, average_time, average_fps, min_fps)

            #cv2.imshow("debug", self.frame)

            self.update_queue(data_car)
            self.iterations += 1

            #print((time.time() - self.start_time) * 1000)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
            normalized_offset = self.center_offset / 800

        #normalized_offset *= -1

        self.angle = np.clip(normalized_offset, -1 + self.steering_offset, 1 - self.steering_offset)

    def calculate_process_time(self):
        """
        Calculate the processing time and FPS (frames per second) statistics.
        
        :return: A tuple containing the following values:
                - total_time: Total processing time in milliseconds
                - fps: Frames per second for the current frame
                - average_time: Average processing time per frame
                - average_fps: Average frames per second over all frames processed
                - min_fps: Minimum frames per second observed
        """
        total_time = (time.time() - self.start_time) * 1000
        fps = 1000 / total_time
        self.total_fps_sum += fps
        if fps < self.min_fps and self.iterations > 50:
            self.min_fps = fps
        self.total_time_sum += total_time
        average_time = self.total_time_sum / self.iterations
        average_fps = self.total_fps_sum / self.iterations
        return total_time, fps, average_time, average_fps, self.min_fps

    def update_queue(self, data_car):
        """
        Update the queue with frames for display and processing.
        :return: None
        """
        steering_wheel = self.draw_steering_wheel()
        orig_frame = cv2.resize(self.frame, (1024, 768))
        detected_lanes = cv2.resize(self.cropped_image, (1024, 768))

        frames = {
        "original": orig_frame,
        "steering": steering_wheel,
        "lanes": detected_lanes
        }
        displayImages(self.four_frame_window, data_car, frames)


    @staticmethod
    def update_data(data, total_time, fps, average_time, average_fps, min_fps):
        """
        Update the data list with processing time and FPS statistics.
        
        :param data: List containing data to be updated
        :param total_time: Total processing time in milliseconds
        :param fps: Frames per second for the current frame
        :param average_time: Average processing time per frame
        :param average_fps: Average frames per second over all frames processed
        :param min_fps: Minimum frames per second observed
        
        :return: Updated data list
        """
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

    def draw_steering_wheel(self):
        """
        Draws a steering wheel image with the specified angle.
        :return: Image of the rotated steering wheel with angle annotation
        """
        # Load the steering wheel image
        steering_wheel_img = cv2.imread('wheel.png', cv2.IMREAD_UNCHANGED)

        height, width = steering_wheel_img.shape[:2]

        # Compute the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), self.angle*90, 1)

        # Rotate the steering wheel image
        rotated_steering_wheel = cv2.warpAffine(steering_wheel_img, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

        # Create a white background image
        white_background = np.ones((768, 1024, 3), dtype=np.uint8) * 255

        # Calculate the position to paste the rotated image
        x_offset = (white_background.shape[1] - rotated_steering_wheel.shape[1]) // 2
        y_offset = (white_background.shape[0] - rotated_steering_wheel.shape[0]) // 2

        # Paste the rotated image onto the white background
        white_background[y_offset:y_offset+rotated_steering_wheel.shape[0], x_offset:x_offset+rotated_steering_wheel.shape[1]] = rotated_steering_wheel

        # Add text annotation
        cv2.putText(white_background, f"Angle: {self.angle}", (20, white_background.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return white_background

    def read_json(self, filename):
        """
        Read JSON data from the specified file.
        :param filename: Path to the JSON file
        :return: Python dictionary containing the JSON data
        """
        with open(filename, 'r') as file:
            data = json.load(file)
        return data

    def detect_changes_in_json(self):
        """
        Detect changes in JSON data and update class attributes accordingly.
        :return: None
        """
        try:
            current_data = self.read_json('data.json')
            self.stream = current_data.get('stream', False)
            self.process = current_data.get('process', False)
            self.motor = current_data.get('motor', False)
            self.debug = current_data.get('debug', False)
            self.placeholder = current_data.get('placeholder', False)
        except:
            pass

class CameraBufferCleanerThread(threading.Thread):
    """
    Thread class for continuously cleaning camera buffer.
    :param camera: OpenCV camera object
    :param name: Name of the thread (default: 'camera-buffer-cleaner-thread')
    """
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        """
        Initialize the thread with the provided camera.
        :param camera: OpenCV camera object
        :param name: Name of the thread (default: 'camera-buffer-cleaner-thread')
        """
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        """
        Run method for the thread.
        Continuously reads frames from the camera and updates the last_frame attribute.
        """
        while True:
            # Read a frame from the camera
            ret, self.last_frame = self.camera.read()


if __name__ == "__main__":
    lane_detector = LaneDetector()
    lane_detector.process_video()