from io import BytesIO
from logging import warning
from traceback import print_exc
from threading import Condition
from PIL import Image
from http.server import BaseHTTPRequestHandler, HTTPServer

# HTML Page content
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

class StreamingServer:
    def __init__(self):
        self.frame = None
        self.server = None
        self.server_running = False

    def start(self):
        try:
            address = ('', 8443)
            self.server = HTTPServer(address, self.StreamingHandler)
            self.server_running = True
            self.server.serve_forever()
        finally:
            self.server_running = False

    def stop_server(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server_running = False

    def update_frame(self, frame):
        self.frame = frame

    def is_running(self):
        return self.server_running

    class StreamingHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def do_GET(self):
            if self.path == '/':
                self.send_response(301)
                self.send_header('Location', '/index.html')
                self.end_headers()
            elif self.path == '/index.html':
                content = PAGE.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Content-Length', len(content))
                self.end_headers()
                self.wfile.write(content)
            elif self.path == '/result.html':
                self.send_header()
                try:
                    stream_path = self.path.split('/')[-1].split('.')[0]
                    self.getStream(stream_path)
                except Exception as e:
                    print_exc()
                    warning(
                        'Removed streaming client %s: %s',
                        self.client_address, str(e))
            else:
                self.send_error(404)
                self.end_headers()

        def sendHeader(self):
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

        def getStream(self, path):
            if path == 'canny_html':
                self.send_frame(self.server.frame)
            elif path == 'field_of_interest_html':
                self.send_frame(self.server.frame)
            elif path == 'detected_lanes_html':
                self.send_frame(self.server.frame)
            elif path == 'result_html':
                self.send_frame(self.server.frame)
            elif path == 'trackline':
                self.send_frame(self.server.frame)
            else:
                self.send_error(404)
                self.end_headers()

        def send_frame(self, frame):
            if frame is not None:
                pil_im = Image.fromarray(frame)
                with BytesIO() as output:
                    pil_im.save(output, "JPEG")
                    frame_data = output.getvalue()
                self.wfile.write(b'--FRAME\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', len(frame_data))
                self.end_headers()
                self.wfile.write(frame_data)
                self.wfile.write(b'\r\n')