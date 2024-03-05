import cv2
import numpy as np

# Load the image
image = cv2.imread('curveroi.png')

cv2.imshow("t7y6zgh",image)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours
for contour in contours:
    # Check if the contour contains enough points
    if len(contour) >= 5:
        # Fit a curve to the contour
        curve = cv2.fitEllipse(contour)
        # Extract parameters of the fitted ellipse
        (x, y), (MA, ma), angle = curve
        
        # Draw the ellipse on the original image
        cv2.ellipse(image, (int(x), int(y)), (int(MA/2), int(ma/2)), angle, 0, 360, (0, 255, 0), 2)
        
        # Calculate length of the curved line (perimeter of ellipse)
        length = cv2.arcLength(contour, True)
        
        # Calculate angle of the curved line
        # In this example, we'll use the angle of the fitted ellipse
        # You can also calculate it based on the slope of tangent lines at different points of the curve
        print("Angle of curved line:", angle)
        print("Length of curved line:", length)
    else:
        print("Contour does not contain enough points.")

# Display the image
cv2.imshow('Curved Line Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
