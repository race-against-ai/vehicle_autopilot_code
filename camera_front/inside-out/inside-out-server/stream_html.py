#import for html website
from io import BytesIO
from logging import warning
from traceback import print_exc
from threading import Condition
from PIL import Image
from http.server import BaseHTTPRequestHandler, HTTPServer
import queue
import time
import datetime

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

        start_time = time.time()

        #send the corresponding stream based on the requested type
        stream_frame = getQueue()
        if path == 'canny_html':
            self.send_frame(stream_frame)
        elif path == 'field_of_interest_html':
            self.send_frame(stream_frame)
        elif path == 'detected_lanes_html':
            self.send_frame(stream_frame)
        elif path == 'result_html':    
            self.send_frame(stream_frame)
        elif path == 'trackline':
            self.send_frame(stream_frame)
        else:
        #handle unknown stream type
            self.send_error(404)
            self.end_headers()

        #print(f"Starting time: {datetime.datetime.now()}. Took {(time.time() - start_time)*1000}")

    def send_frame(self, frame):
        #convert frame to PIL Image
        if frame is not None:
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

image_queue = queue.Queue(maxsize=1)

def isFull():
    if image_queue.full() == True:
        return True
    else:
        return False
    
def putQueue(item):
    if not isFull():
        image_queue.put(item)

def getQueue():
    return image_queue.get(block=True)

if __name__ == "__main__":
    init()
