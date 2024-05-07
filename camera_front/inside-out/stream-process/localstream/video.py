import threading
from stream_html import init, putQueue
import cv2


def video():

    # URL of the video stream served by the HTTP server
    stream_url = "output_video.mp4"

    # Create a VideoCapture object to access the stream
    cap = cv2.VideoCapture(stream_url)

    # Check if the capture is successful
    if not cap.isOpened():
        print("Error: Unable to open video stream")
        exit()

    # Read and display frames from the stream
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to retrieve frame from stream")
            break

        cv2.imshow("video", frame)
        putQueue(frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #start the backround-thread for the frame
    recv_thread = threading.Thread(target=video)
    stream_thread = threading.Thread(target=init)
    stream_thread.start()
    recv_thread.start()