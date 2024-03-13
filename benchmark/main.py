import time

import cv2
import numpy as np

from pathlib import Path


FILE_PATH = Path(__file__).parent

VIDEO_PATH = FILE_PATH / "../camera_front/inside-out/inside-out-video/videos/no-off.avi"


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    def time_passed_ms(self, reset=True) -> int:
        current_time = time.time()
        delta = int((current_time - self.start_time) * 1000)
        if reset:
            self.start_time = current_time
        return delta

    def print_time_passed_ms(self, prefix: str = "", reset=True):
        time_passed_ms = self.time_passed_ms(reset)
        print(f"{prefix} {time_passed_ms}")





if __name__ == "__main__":

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    timer = Timer()

    while True:

        timer.reset()
        _, original_image = cap.read()
        draw_image = original_image.copy()
        height, width, depth = original_image.shape
        timer.print_time_passed_ms("Capture:")

        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        thresh, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        timer.print_time_passed_ms("Preprocessing:")

        search_y = 300
        results = np.where(binary_image[search_y, :] == 255)

        timer.print_time_passed_ms("searching:")

        cv2.line(draw_image,
                 (0, search_y),
                 (width, search_y),
                 (0, 0, 255),
                 5)

        for i in range(len(results[0])):
            x = results[0][i]
            y = search_y
            cv2.circle(draw_image, (x, y), 3, (255, 0, 0))

        timer.print_time_passed_ms("Drawing:")

        # cv2.imshow('binary_image', binary_image)
        # cv2.imshow('draw_image', draw_image)
        #
        #
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break