import cv2
import numpy as np
import socket
import time


def send_images():
    # Initialize socket
    cap = cv2.VideoCapture(0)
    sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    receiver_address = ('192.168.50.1', 12345)

    try:
        sender_socket.connect(receiver_address)
    except Exception as e:
        print("Error:", e)


    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1024, 768))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        try:

            start_time = time.time()

            # Capture or load image
            image = frame  # Load image from file or capture using cv2.VideoCapture()
            # Encode image to bytes
            encoded_image = image.tobytes()
            # Send image size first
            image_size = len(encoded_image)
            sender_socket.sendall(image_size.to_bytes(4, byteorder='big'))

            # Send image data
            sender_socket.sendall(encoded_image)

            print("send time")
            print((time.time() - start_time) * 1000)

            response_time = time.time()
            # Wait for response from receiver
            response = sender_socket.recv(1024)
            print("Response from receiver:", response.decode())
            print((time.time() - response_time) * 1000)



        except Exception as e:
            sender_socket.close()
            sender_socket.connect(receiver_address)
            print("Error:", e)


send_images()
