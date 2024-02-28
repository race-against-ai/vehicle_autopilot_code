from driving import Functions_Driving
driving_instance = Functions_Driving()

from io import StringIO, BytesIO
from logging import warning
from traceback import print_exc
from threading import Condition
from PIL import ImageFont, ImageDraw, Image
from http.server import BaseHTTPRequestHandler, HTTPServer
from piracer.cameras import MonochromeCamera
from imutils import rotate


# Importing required OpenCV modules
from cv2 import COLOR_RGB2BGR, cvtColor

# Importing MonochromeCamera from piracer.cameras
from piracer.cameras import MonochromeCamera

# Importing the lane_pi module
from main import main_detect

# Importing the imutils module
from imutils import rotate

# Define the second camera
camera = MonochromeCamera()  # You might need to adjust this based on your actual camera initialization

PAGE = """\
<!DOCTYPE html>
<html>
<head>
    <title>Video Stream</title>
    <style>
        body {
            margin: 20px;
        }

        h1 {
            text-align: center;
        }

        .stream-container {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px; /* Adjust the gap as needed */
        }

        

        .stream-title {
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>PiRacer video Stream</h1>
    <div class="stream-container">
        <div>
            <p class="stream-title">Frame</p>
            <img class="stream" src="stream.mjpg" alt="Stream 1">
        </div>
        <div>
            <p class="stream-title">Result on birds Eye</p>
            <img class="stream" src="stream.mjpg" alt="Stream 2">
        </div>
        <div>
            <p class="stream-title">Result on Original</p>
            <img class="stream" src="stream.mjpg" alt="Stream 3">
        </div>
        <div>
            <p class="stream-title">Result sliding Windows</p>
            <img class="stream" src="stream.mjpg" alt="Stream 4">
        </div>
    </div>
</body>
</html>
"""

# Class to manage streaming output
class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = BytesIO()
        self.condition = Condition()

    def write(self, buf, camera_id=1):
        if buf.startswith(b'\xff\xd8'):
            if camera_id == 1:
                # Clear buffer for camera 1
                self.buffer.truncate()
                with self.condition:
                    self.frame = self.buffer.getvalue()
                    self.condition.notify_all()
                self.buffer.seek(0)
            elif camera_id == 2:
                # Clear buffer for camera 2
                self.buffer2.truncate()
                with self.condition:
                    self.frame2 = self.buffer2.getvalue()
                    self.condition.notify_all()
                self.buffer2.seek(0)
        if camera_id == 1:
            # Write buffer for camera 1
            return self.buffer.write(buf)
        elif camera_id == 2:
            # Write buffer for camera 2
            return self.buffer2.write(buf)

# Create StreamingOutput instance
output = StreamingOutput()

# Class to handle HTTP requests
class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Redirect root path to index.html
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            # Serve the HTML page
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            # MJPEG streaming path
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                
                while True:


                    #user_input = input("Enter a command ('start' to run, 'exit' to leave): ")
                    #if user_input.lower() == "start":
                    driving_instance.frward_drive()
                    
                    #elif user_input.lower() == "stop":
                        #print("Stopping the program.")
                        #driving_instance.breaking()

                    # Read a frame from the camera
                    frame = camera.read_image()

                    # Apply main_detect function to get different video streams
                    normal_video, birds_eye, original_masked, sliding_video = main_detect(frame)

                    # Send the first stream
                    self.send_frame(normal_video)

                    # Send the second stream
                    self.send_frame(normal_video)

                    # Send the third stream
                    self.send_frame(normal_video)

                    # Send the fourth stream
                    self.send_frame(normal_video)

            except Exception as e:
                print_exc()
                warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            # Handle 404 error
            self.send_error(404)
            self.end_headers()

    def send_frame(self, frame):
        # Convert frame to PIL Image
        pil_im = Image.fromarray(frame)

        # Convert Image to JPEG format and get the frame data
        with BytesIO() as output:
            pil_im.save(output, "JPEG")
            frame_data = output.getvalue()

        # Write the frame data to the client
        self.wfile.write(b'--FRAME\r\n')
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', len(frame_data))
        self.end_headers()
        self.wfile.write(frame_data)
        self.wfile.write(b'\r\n')

#class to handle StereamingServer
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