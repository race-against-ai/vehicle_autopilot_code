import cv2
import time

# Function to process the image
def process_image(img):
    # Example processing: Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

# Main function
def main():
    # Open the default camera (usually the webcam)
    cap = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return
    
    while True:
        start_time = time.time()
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret:
            # Measure start time
            
            # Process the image
            processed_frame = process_image(frame)
            
            # Measure end time
            end_time = time.time()
            
            # Calculate processing time
            processing_time = end_time - start_time
            
            # Print processing time
            print("Processing time: {:.2f} seconds".format(processing_time))
            
if __name__ == "__main__":
    main()
