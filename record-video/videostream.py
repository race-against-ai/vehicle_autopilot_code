import cv2

def start_recording():
    # Open the default camera (ID 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera was successfully opened
    if not cap.isOpened():
        print("Error opening the camera.")
        return

    # Define the frame rate and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1024, 768))

    while True:
        # Capture a frame
        ret, frame = cap.read()

        if not ret:
            print("Error reading the frame.")
            break

    # Release resources
    cap.release()
    out.release()

def main():
    print("Commands:")
    print("start - Start video recording")
    print("stop - Stop the recording")

    while True:
        command = input("Enter command: ")

        if command == "start":
            start_recording()
        elif command == "stop":
            break
        else:
            print("Invalid command. Try again.")

if __name__ == "__main__":
    main()
