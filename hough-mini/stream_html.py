#camera import

#colored frames
#from color_camera import Camera
#camera = Camera()

#black-white frame
from piracer.cameras import Camera, MonochromeCamera
camera = MonochromeCamera()

#import for steering left and right
from driving import Functions_Driving
driving_instance = Functions_Driving()

#import for angle
from trackline import Trackline
trackline_import = Trackline()

#import line detection
from hough_lanes import main_lanes

#import for html website
from io import StringIO, BytesIO
import cv2
from logging import warning
from traceback import print_exc
from threading import Condition
from PIL import ImageFont, ImageDraw, Image
from http.server import BaseHTTPRequestHandler, HTTPServer
import copy
import time
from prettytable import PrettyTable
import os

#importing required OpenCV modules
from cv2 import COLOR_RGB2BGR, cvtColor


driving_instance.battery_percent()

track = 0
centroids = 0
backup_centroids = 0

def main(firstRun = False):
    global track, centroids, backup_centroids

    #read a frame from the camera
    frame = camera.read_image()

    start_time = time.time()

    if firstRun:
        #get tracking points
        track, centroids, backup_centroids = getTrackline(frame)

    #apply main_detect function to get different video streams
    data = main_lanes(frame, centroids, backup_centroids)

    total_time = (time.time() - start_time) * 1000  #Total time

    fps = 1000/total_time
    data.append(total_time)
    data.append(fps)

    #show battery percentage on display
    #driving_instance.battery_percent()

    #driving_instance.frward_drive(0.2)

    print_table(data)

        

def getTrackline(frame):

    #define the coordinates and size of the squares
    square1 = (0, 0, 500, 770)  # (x, y, width, height)
    square2 = (550, 0, 514, 770) #rechts
    square3 = (0, 635, 1024, 180) #unten
    square4 = (0, 0, 1024, 570) #oben
    
    test = copy.deepcopy(frame)

    #draw the filled squares on the black image
    cv2.rectangle(test, (square1[0], square1[1]), (square1[0] + square1[2], square1[1] + square1[3]), color=(0, 0, 0), thickness=cv2.FILLED)
    cv2.rectangle(test, (square2[0], square2[1]), (square2[0] + square2[2], square2[1] + square2[3]), color=(0, 0, 0), thickness=cv2.FILLED)
    cv2.rectangle(test, (square3[0], square3[1]), (square3[0] + square3[2], square3[1] + square3[3]), color=(0, 0, 0), thickness=cv2.FILLED)
    cv2.rectangle(test, (square4[0], square4[1]), (square4[0] + square4[2], square4[1] + square4[3]), color=(0, 0, 0), thickness=cv2.FILLED)

    #apply main_detect function to get different video streams
    track, centroids, backup_centroids  = trackline_import.run(test)

    return track, centroids, backup_centroids

def print_table(data):
    # Erstellen der Tabelle
    table = PrettyTable()
    table.field_names = ["Timestamp","Canny Image", "Region of Interest","Hough Lines", "Line Image", "Total Image Time", "Total Main Loop", "FPS"]
    table.add_row(data)
    
    # Ausgabe der Tabelle
    print(table)


if __name__ == "__main__":
    main(True)
    while True:
        main()
