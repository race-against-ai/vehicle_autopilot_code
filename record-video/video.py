import cv2

# Open the camera
video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera, you can change it to other indexes if you have multiple cameras

# Check if the camera opened successfully
if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the camera's frame width, height, and frames per second
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for saving video (XVID for .avi)
out = cv2.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))


# Loop through each frame from the camera
while video_capture.isOpened():
    ret, frame = video_capture.read()  # Read the next frame from the camera
    if not ret:
        break  # Break the loop if there are no more frames

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Process the frame here if needed (e.g., apply filters, annotations, etc.)

    # Write the frame to the output video
    out.write(frame)

# Release resources
video_capture.release()
out.release()
