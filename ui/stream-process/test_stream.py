import cv2

video_path = 'http://192.168.30.106:8443/result_html.py'

# Initialize the video stream
cap = cv2.VideoCapture(video_path) 

# Read a frame from the video stream
ret, frame = cap.read()

# Resize the frame
resized_frame = cv2.resize(frame, (1024, 768))

# Display the resized frame
cv2.imshow('Resized Frame', resized_frame)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
