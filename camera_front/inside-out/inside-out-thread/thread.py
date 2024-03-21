import threading
import time
import cv2
from main import LaneDetector
from stream_html import update_frame
from piracer.cameras import Camera, MonochromeCamera
camera = MonochromeCamera()

shared_frame = None
mutex = threading.Lock()

lane_detector = LaneDetector()

#function for getting camera frame
def update_frame():
    global shared_frame
    while True:
        with mutex:
            frame = camera.read_image()
            # cap = cv2.VideoCapture(0)
            # ret, frame = cap.read()
            frame = cv2.resize(frame, (1024, 768))
            print(frame)
            shared_frame = frame


def thread1_function():
    global shared_frame
    while True:
        with mutex:
            local_copy = shared_frame
            
            thread1_start_time = time.time()
            lane_detector.process_video(local_copy)
            thread1_time = (time.time() - thread1_start_time) * 1000
            print(f"thread1: {thread1_time}ms")

def thread2_function():
    global shared_frame
    while True:
        with mutex:
            local_copy = shared_frame
            thread2_start_time = time.time()
            update_frame(local_copy)
            thread2_time = (time.time() - thread2_start_time) * 1000
            print(f"thread2: {thread2_time}ms")

if __name__ == "__main__":
    #start the backround-thread for the frame
    update_thread = threading.Thread(target=update_frame)

    #threads for lanedetection and stream
    thread1 = threading.Thread(target=thread1_function)
    thread2 = threading.Thread(target=thread2_function)
    update_thread.start()
    thread1.start()
    thread2.start()
