import time
import numpy as np
from prettytable import PrettyTable
import json
from lane_detection import TimerClass, call_lane_detection
from driving import Functions_Driving
from numpy.typing import NDArray

driving_instance = Functions_Driving()

def time_tracker(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        if not hasattr(self, 'time_data'):
            self.time_data = []
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        self.time_data.append(duration)
        return result
    return wrapper

class FrameProcess:
    def __init__(self):
        self.frame : NDArray[np.float64]  = None
        dummy_frame : NDArray[np.float64]  = np.zeros((720, 960, 3), dtype=np.uint8)
        self.LaneDetectionInstance : TimerClass = TimerClass()
        call_lane_detection(self.LaneDetectionInstance, dummy_frame)
        self.center_offset : float = None
        self.data : list[float] = None
        self.curve : bool = None
        self.speed_curve : bool = None

        self.stream : bool = None
        self.process : bool = None
        self.motor : bool = None
        self.debug : bool = None
        self.placeholde : bool = None

        self.steering_offset : float = 0.21

    def set_frame(self, frame):
        self.frame = frame

    def get_lane_infos(self):
        self.center_offset, self.data, self.curve, self.speed_curve = call_lane_detection(self.LaneDetectionInstance, self.frame)

    def get_process_time(self, full_time):
        self.data.append(full_time[0] + full_time[1] + full_time[2])
        try:
            fps = 1000 / full_time[0] + full_time[1] + full_time[2]
        except:
            fps = 1
        self.data.append(fps)
        self.data = [round(num, 3) if isinstance(num, float) else num for num in self.data]
        table = PrettyTable()
        table.field_names = ["Rows selection", "Binary", "Nearest Pixel", "Curve", "Brake", "Steering Angle", "Total Main Loop", "FPS"]  #, "Cam", "Total Main Loop", "FPS", "Average Time", "Average FPS", "min FPS"]
        table.add_row(self.data)
        if self.process:
            print(table)

    def get_json(self):
        try:
            current_data = self.read_json('data.json')
            self.stream = current_data.get('stream', False)
            self.process = current_data.get('process', False)
            self.motor = current_data.get('motor', False)
            self.debug = current_data.get('debug', False)
            self.placeholder = current_data.get('placeholder', False)
        except Exception as e:
            print(f"Failed to read JSON data: {e}")

    @staticmethod
    def read_json(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        return data

class MotorControl(FrameProcess):
    def __init__(self):
        super().__init__()
        self.angle : float = 0
        self.tester : bool = False
        self.last_curve_tester : bool = False
        self.curve_tester : int = None
        self.brake : int = 0

    def calculate_steering_angle(self):
        if self.curve:
            normalized_offset = self.center_offset / 350
        else:
            normalized_offset = self.center_offset / 1000
        self.angle = np.clip(normalized_offset, -1, 1)

    def motor_control(self):
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
                if self.brake == 0:
                    self.brake += 1
                elif self.brake < 2: #change here how often the card should brake
                        driving_instance.frward_drive(0)
                        self.brake += 1
                else:
                    #print("--------------------------------------------------------------------------")
                    driving_instance.frward_drive(0.35) #0.4
            else:
                self.brake = 0
                #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                driving_instance.frward_drive(0.4) #0.56
        else:
            driving_instance.frward_drive(0)


    def steering_control(self):
        driving_instance.left_steering(np.clip(self.angle - self.steering_offset, -1, 1))

class MainHandler(MotorControl):
    def __init__(self):
        MotorControl.__init__(self)
        self.time_data : list[float] = []

    @time_tracker
    def get_lane_infos(self):
        super().get_lane_infos()

    @time_tracker
    def motor_control(self):
        super().motor_control()

    @time_tracker
    def steering_control(self):
        super().steering_control()

def call_all_methods(MainInstance, frame):

    MainInstance.set_frame(frame)
    MainInstance.get_lane_infos()
    MainInstance.get_json()
    MainInstance.calculate_steering_angle()
    MainInstance.motor_control()
    MainInstance.steering_control()
    MainInstance.get_process_time(MainInstance.time_data)
    MainInstance.time_data = []

    print(f'Kurve {MainInstance.curve}, {MainInstance.speed_curve}')
    # print(MainInstance.angle)
    print(MainInstance.LaneDetectionInstance.curve_direction)