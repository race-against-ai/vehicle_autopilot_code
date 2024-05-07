import socket


def receive_images(save_path, host, port):
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
            # Receive the image size
            image_size_bytes = conn.recv(8)
            if not image_size_bytes:
                break

            image_size = int.from_bytes(image_size_bytes, byteorder='big')

            # Receive the image data
            with open(save_path, 'wb') as f:
                remaining = image_size
                while remaining > 0:
                    data = conn.recv(1024)
                    if not data:
                        break
                    f.write(data)
                    remaining -= len(data)

            print("Image received successfully")

if __name__ == "__main__":
    # Path to save the received image
    save_path = "received_image.jpg"

    # Receiver's IP address and port
    host = '0.0.0.0'
    port = 12345

    receive_images(save_path, host, port)
