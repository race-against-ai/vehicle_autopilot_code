import threading
import cv2

# Define the thread that will continuously pull frames from the camera
class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()


def cam():
    # Start the camera
    stream_url = "http://localhost:8443/canny_html.py"
    camera = cv2.VideoCapture(stream_url)

    # Start the cleaning thread
    cam_cleaner = CameraBufferCleanerThread(camera)

    # Use the frame whenever you want
    while True:
        if cam_cleaner.last_frame is not None:
            cv2.imshow('The last frame', cam_cleaner.last_frame)
        cv2.waitKey(10)