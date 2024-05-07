import socket
import os
import time

def send_image(image_path, host, port):
    # Create a TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Connect to the receiver
        s.connect((host, port))

        while True:
            # Send the image file size first
            image_size = os.path.getsize(image_path)
            s.sendall(image_size.to_bytes(8, byteorder='big'))

            # Send the image file
            with open(image_path, 'rb') as f:
                while True:
                    data = f.read(1024)
                    if not data:
                        break
                    s.sendall(data)

            print("Image sent successfully")
            time.sleep(1)  # Wait for 1 second before sending again

if __name__ == "__main__":
    # Image path
    image_path = "image.png"

    # Receiver's IP address and port
    host = '192.168.50.1'
    port = 12345

    while True:
        send_image(image_path, host, port)
