import cv2

# Path to the video file
video_path = 'no-off.avi'

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    frame = cv2.resize(frame, (1024, 768))
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    height, width = frame.shape[:2]

    if len(frame.shape) == 3:
        num_channels = frame.shape[2]
        print(f"The image has {num_channels} channels (color image)")
    elif len(frame.shape) == 2:
        print("The image is grayscale (1 channel)")
    else:
        print("Unsupported image format")


    # Check if the frame was successfully read
    if not ret:
        break

    # Display the frame
    cv2.imshow('Frame', frame)

    # Check for the 'q' key to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close any open windows
cap.release()
cv2.destroyAllWindows()
