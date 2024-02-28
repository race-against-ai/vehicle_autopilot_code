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
from sliding_lane import main_detect
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


#importing required OpenCV modules
from cv2 import COLOR_RGB2BGR, cvtColor


#html website
PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pi-Car-Turbo Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }
        header {
            background-color: #333;
            padding: 10px;
            text-align: center;
            color: white;
        }
        nav {
            background-color: #eee;
            padding: 10px;
        }
        nav a {
            margin-right: 15px;
            text-decoration: none;
            color: #333;
            font-weight: bold;
        }
        section {
            padding: 20px;
        }
    </style>
</head>
<body>
    <header>
        <h1>Pi-Car-Turbo Stream</h1>
    </header>
    <nav>
        <a href="/canny_html.py">Canny</a>
        <a href="/field_of_interest_html.py">Field of Interest</a>
        <a href="/detected_lanes_html.py">Hough Lanes</a>
        <a href="/result_html.py">Result</a>
        <a href="/trackline.py">Tracking</a>
        <a href="/normal.py">Normal</a>
    </nav>
    <section>
        <h2>Welcome to Pi-Car-Turbo Stream!</h2>
        <p>This is a simple website with Python backend.</p>
    </section>
</body>
</html>
"""


#class to manage streaming output
class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = BytesIO()
        self.condition = Condition()

    def write(self, buf, camera_id=1):
        if buf.startswith(b'\xff\xd8'):
            if camera_id == 1:
                #clear buffer for camera 1
                self.buffer.truncate()
                with self.condition:
                    self.frame = self.buffer.getvalue()
                    self.condition.notify_all()
                self.buffer.seek(0)
            elif camera_id == 2:
                #clear buffer for camera 2
                self.buffer2.truncate()
                with self.condition:
                    self.frame2 = self.buffer2.getvalue()
                    self.condition.notify_all()
                self.buffer2.seek(0)
        if camera_id == 1:
            #write buffer for camera 1
            return self.buffer.write(buf)
        elif camera_id == 2:
            #write buffer for camera 2
            return self.buffer2.write(buf)

#create StreamingOutput instance
output = StreamingOutput()

#class to handle HTTP requests
class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            #redirect root path to index.html
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            #serve the HTML page
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path.startswith('/canny_html.py'):
            
            self.sendHeader()

            try:
                while True:
                    #identify the requested stream type based on the path
                    stream_path = self.path.split('/')[-1].split('.')[0]

                    #apply main_detect function to get different video streams
                    self.getStream(stream_path)

            except Exception as e:
                print_exc()
                warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
                

        elif self.path.startswith('/field_of_interest_html.py'):
            
            self.sendHeader()

            try:
                while True:

                    #identify the requested stream type based on the path
                    stream_path = self.path.split('/')[-1].split('.')[0]

                    #apply main_detect function to get different video streams
                    self.getStream(stream_path)


            except Exception as e:
                print_exc()
                warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
    
        elif self.path.startswith('/detected_lanes_html.py'):
            
            self.sendHeader()

            try:
                while True:

                    #identify the requested stream type based on the path
                    stream_path = self.path.split('/')[-1].split('.')[0]

                    #apply main_detect function to get different video streams
                    self.getStream(stream_path)

            except Exception as e:
                print_exc()
                warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
                
                
        elif self.path.startswith('/result_html.py'):
            
            self.sendHeader()

            try:
                while True:

                    #identify the requested stream type based on the path
                    stream_path = self.path.split('/')[-1].split('.')[0]

                    #apply main_detect function to get different video streams
                    self.getStream(stream_path)

                    #driving_instance.frward_drive(0.19)

            except Exception as e:
                print_exc()
                warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
                
                
        elif self.path.startswith('/trackline.py'):
            
            self.sendHeader()

            try:
                while True:
                   
                    #identify the requested stream type based on the path
                    stream_path = self.path.split('/')[-1].split('.')[0]

                    #apply main_detect function to get different video streams
                    self.getStream(stream_path)
 
            except Exception as e:
                print_exc()
                warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
                

        elif self.path.startswith('/normal.py'):
            
            self.sendHeader()

            try:
                while True:
                    #read a frame from the camera
                    frame = camera.read_image()

                    #identify the requested stream type based on the path
                    stream_type = self.path.split('/')[-1].split('.')[0]
                    #frame = cvtColor(frame, COLOR_RGB2BGR)

                    driving_instance.battery_percent()

                    #send the corresponding stream based on the requested type
                    if stream_type == 'normal':
                        self.send_frame(frame)
                    else:
                        #handle unknown stream type
                        self.send_error(404)
                        self.end_headers()
                        break
 
            except Exception as e:
                print_exc()
                warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))

                
        else:
            #handle 404 error
            self.send_error(404)
            self.end_headers()


    def sendHeader(self):

        #MJPEG streaming path
        self.send_response(200)
        self.send_header('Age', 0)
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.end_headers()

    def getStream(self, path):

        #read a frame from the camera
        frame = camera.read_image()

        #get tracking points
        track, centroids, backup_centroids = self.getTrackline(frame)

        #apply main_detect function to get different video streams
        canny, field_of_interest, detected_lanes, result = main_lanes(frame, centroids, backup_centroids)

        #show battery percentage on display
        driving_instance.battery_percent()

        #send the corresponding stream based on the requested type
        if path == 'canny_html':
            self.send_frame(canny)
        elif path == 'field_of_interest_html':
            self.send_frame(field_of_interest)
        elif path == 'detected_lanes_html':
            self.send_frame(detected_lanes)
        elif path == 'result_html':
            
            self.send_frame(result)
        elif path == 'trackline':
            self.send_frame(track)
        else:
        #handle unknown stream type
            self.send_error(404)
            self.end_headers()
        

    def getTrackline(self, frame):

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

    def send_frame(self, frame):
        #convert frame to PIL Image
        pil_im = Image.fromarray(frame)

        #convert Image to JPEG format and get the frame data
        with BytesIO() as output:
            pil_im.save(output, "JPEG")
            frame_data = output.getvalue()

        #write the frame data to the client
        self.wfile.write(b'--FRAME\r\n')
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', len(frame_data))
        self.end_headers()
        self.wfile.write(frame_data)
        self.wfile.write(b'\r\n')


#class to handle StreamingServer
class StreamingServer(HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def init():
    try:
        address = ('', 8443)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        pass

if __name__ == "__main__":
    init()
