import multiprocessing
import numpy as np
import cv2
import time
from main import LaneDetector

def capture_video(shared_array, event, shape, dtype):
    url = 'output_video.avi'
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0*30 )
    

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
        
        if frame.shape != shape:
            print(f"Error: Frame shape {frame.shape} does not match the expected shape {shape}.")
            break

        event.wait()
        event.clear()
        with shared_array.get_lock():
            np_array = np.frombuffer(shared_array.get_obj(), dtype=dtype).reshape(shape)
            np.copyto(np_array, frame)
        
        fps_counter += 1
        if time.time() - start_time >= 1.0:
            #print(f"Capture FPS: {fps_counter}")
            fps_counter = 0
            start_time = time.time()

    cap.release()

def process_detection(shared_array, event, shape, dtype):
    fps_counter = 0
    start_time = time.time()
    lanedetector = LaneDetector()
    while True:
        with shared_array.get_lock():
            np_array = np.frombuffer(shared_array.get_obj(), dtype=dtype).reshape(shape)
            frame = np_array.copy()

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        lanedetector.process_video(frame)

        event.set()
        
        fps_counter += 1
        if time.time() - start_time >= 1.0:
            fps_counter = 0
            start_time = time.time()


if __name__ == '__main__':
    dummy_frame = np.zeros((720, 960, 3), dtype=np.uint8)
    shape = dummy_frame.shape
    dtype = dummy_frame.dtype

    shared_array = multiprocessing.Array(dtype.char, dummy_frame.size)
    event = multiprocessing.Event()

    process_capture = multiprocessing.Process(target=capture_video, args=(shared_array, event, shape, dtype))
    process_lane_detection = multiprocessing.Process(target=process_detection, args=(shared_array, event, shape, dtype))

    process_capture.start()
    process_lane_detection.start()