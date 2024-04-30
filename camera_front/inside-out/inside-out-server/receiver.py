import socket
from stream_html import init, putQueue
import threading
import io
from PIL import Image
import numpy as np
import cv2

def receive_image_data(conn):
    image_data = b''  # Initialize an empty byte string to store image data
    remaining = 0

    # Receive the image size
    image_size_bytes = conn.recv(8)
    if not image_size_bytes:
        return None  # If no data received, return None

    image_size = int.from_bytes(image_size_bytes, byteorder='big')

    # Receive the image data
    while remaining < image_size:
        data = conn.recv(1024)
        if not data:
            break
        image_data += data
        remaining += len(data)

    return image_data

def receive_data():
    # Receiver's IP address and port
    host = '0.0.0.0'
    port = 12345

    # Create a TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind the socket to the host and port
        s.bind((host, port))

        # Listen for incoming connections
        s.listen()

        print("Waiting for connection...")

        # Accept a connection
        conn, addr = s.accept()

        print(f"Connected to {addr}")

        while True:
            # Receive image data
            image_data = receive_image_data(conn)
            if image_data is None:
                print("No Image")
                break  # Break if no data received

            print("Received Image")

                    # Convert BytesIO object to PIL image
            # Convert BytesIO object to PIL image
            try:
                pil_im = Image.open(io.BytesIO(image_data))
            except:
                pil_im = None
                break
            
            # Convert PIL image to NumPy array
            numpy_array = np.array(pil_im)

            # Process the image data here
            # For demonstration, let's just print the length of the received image data
            putQueue(numpy_array)


if __name__ == "__main__":
    main_thread = threading.Thread(target=receive_data)
    stream_thread = threading.Thread(target=init)
    main_thread.start()
    stream_thread.start()
