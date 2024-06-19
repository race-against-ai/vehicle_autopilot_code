import multiprocessing
import numpy as np
import cv2
import time
import os
from main import MainHandler, call_all_methods
from stream_html import init, putQueue
import threading

#set camera filter and rotation at start
device_path='/dev/video0'
os.system(f'v4l2-ctl -d {device_path} --set-ctrl=rotate={180}')
os.system(f'v4l2-ctl -d {device_path} --set-ctrl=color_effects=1') # Run 'v4l2-ctl -L' for explanations

def capture_video(shared_array, event, shape, dtype):
    # Initialize the video capture
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    fps_counter = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Ensure the frame fits in the shared array
        if frame.shape != shape:
            print(f"Error: Frame shape {frame.shape} does not match the expected shape {shape}.")
            break

        # Copy the frame to the shared array
        with shared_array.get_lock():
            np_array = np.frombuffer(shared_array.get_obj(), dtype=dtype).reshape(shape)
            np.copyto(np_array, frame)
        event.set()
        
        fps_counter += 1
        if time.time() - start_time >= 1.0:
            #print(f"Capture FPS: {fps_counter}")
            fps_counter = 0
            start_time = time.time()

    cap.release()

def process_detection(shared_array, processed_image, shape, dtype):
    fps_counter = 0
    start_time = time.time()
    mainhandler = MainHandler()

    while True:
        # Convert the shared array back to a numpy array

        with shared_array.get_lock():
            np_array = np.frombuffer(shared_array.get_obj(), dtype=dtype).reshape(shape)
            frame = np_array.copy()

        # Process the frame using the lane detector
        call_all_methods(mainhandler, frame)


        fps_counter += 1
        if time.time() - start_time >= 1.0:
            #print(f"Lane Detection FPS: {fps_counter}")
            fps_counter = 0
            start_time = time.time()

def process_stream(shared_array, event, shape, dtype):
    fps_counter = 0
    start_time = time.time()

    # Start init() in a separate thread
    init_thread = threading.Thread(target=init)
    init_thread.daemon = True  # Allow thread to exit when the main program exits
    init_thread.start()

    while True:
        # wait for event on video process and clear it
        event.wait()
        event.clear()

        # Convert the shared array back to a numpy array
        with shared_array.get_lock():
            np_array = np.frombuffer(shared_array.get_obj(), dtype=dtype).reshape(shape)
            frame = np_array.copy()

        # Put the numpy array into the queue
        putQueue(frame)
        
        # Update FPS counter
        fps_counter += 1
        if time.time() - start_time >= 1.0:
            #print(f"Stream FPS: {fps_counter}")
            fps_counter = 0
            start_time = time.time()

if __name__ == '__main__':
    # Define the frame properties
    dummy_frame = np.zeros((720, 960, 3), dtype=np.uint8)  # Assuming 640x480 RGB frames
    shape = dummy_frame.shape
    dtype = dummy_frame.dtype

    # Create a shared memory array
    shared_array = multiprocessing.Array(dtype.char, dummy_frame.size)
    processed_image = multiprocessing.Array(dtype.char, dummy_frame.size)
    event = multiprocessing.Event()

    # Create the processes
    process_capture = multiprocessing.Process(target=capture_video, args=(shared_array, event, shape, dtype))
    process_lane_detection = multiprocessing.Process(target=process_detection, args=(shared_array, processed_image, shape, dtype))
    process_stream_wlan = multiprocessing.Process(target=process_stream, args=(shared_array, processed_image, event, shape, dtype))

    # Start the processes
    process_capture.start()
    process_lane_detection.start()
    process_stream_wlan.start()
