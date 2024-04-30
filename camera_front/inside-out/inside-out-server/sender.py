import socket
import os
import time
import queue
import numpy as np

def send_image(host, port):
    # Create a TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Connect to the receiver
        s.connect((host, port))

        while True:
            # Convert the image array to bytes
            image_array = getQueue()
            image_bytes = image_array.tobytes()

            # Send the image size first
            image_size = len(image_bytes)
            s.sendall(image_size.to_bytes(8, byteorder='big'))

            # Send the image bytes
            s.sendall(image_bytes)

            print("Image sent successfully")

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

def sender_frame():
    # Receiver's IP address and port
    host = '192.168.50.1'
    port = 12345

    send_image(host, port)

if __name__ == "__main__":
    sender_frame()
