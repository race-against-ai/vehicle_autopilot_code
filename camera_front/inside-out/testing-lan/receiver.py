import cv2
import socket
import numpy as np
import threading

def recv():
    # Receiver

    # Initialize socket
    receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the host and port
    host = '192.168.50.1'
    port = 12345
    receiver_socket.bind((host, port))

    # Listen for incoming connections
    receiver_socket.listen(1)
    print("Waiting for connection...")

    # Accept a connection
    connection, sender_address = receiver_socket.accept()
    print("Connection established with:", sender_address)

    while True:
        try:
            # Receive image size
            size_bytes = connection.recv(4)
            image_size = int.from_bytes(size_bytes, byteorder='big')

            connection.sendall(b"Image received successfully.")

            # Receive image data
            received_data = b''
            while len(received_data) < image_size:
                data = connection.recv(image_size - len(received_data))
                if not data:
                    break
                received_data += data

            # Send response

            import numpy as np
            # Example values
            original_shape = (768, 1024)  # Replace height, width, and channels with actual values
            dtype = np.uint8  # Replace with the actual data type of the original imag

            decoded_image = np.frombuffer(received_data, dtype=dtype).reshape(original_shape)

            cv2.imshow("prew", decoded_image)
        except Exception as e:
            receiver_socket.close()
            print("Error:", e)
            break

    recv()


if __name__ == "__main__":
    #start the backround-thread for the frame
    recv_thread = threading.Thread(target=recv)
    recv_thread.start()