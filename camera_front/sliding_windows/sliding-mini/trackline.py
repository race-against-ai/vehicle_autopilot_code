import cv2
import numpy as np

#initialize global backup centroids with default values
backup_centroids = [(0, 0), (0, 0)]

class Trackline:
    def __init__(self):
        #initialize last centroids to None
        self.last_centroid1 = None
        self.last_centroid2 = None

    def get_thresholded_img(self, im):
        #convert the input image to grayscale
        #imghsv = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #threshold the image to obtain a binary mask
        img_threshold = cv2.inRange(im, 200, 255)  # Adjust the threshold values as needed
        return img_threshold

    def get_positions(self, im):
        #access the global backup centroids
        global backup_centroids
        
        #find contours in the thresholded image
        contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []

        #calculate centroids for each contour
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
            
        #update backup centroids if at least two centroids are found, otherwise use the backup
        if len(centroids) >= 2:
            backup_centroids = centroids
        else:
            centroids = backup_centroids

        return centroids, backup_centroids

    def process_frame(self, test):
        #initialize an image for drawing
        imdraw = np.zeros_like(test)
        
        # Get the thresholded image
        imgyellowthresh = self.get_thresholded_img(test)
        
        #apply erosion to the thresholded image
        kernel = np.ones((5, 5), np.uint8)
        imgyellowthresh = cv2.erode(imgyellowthresh, kernel, iterations=1)
        
        #get current and backup centroids
        centroids, backup_centroids = self.get_positions(imgyellowthresh)

        #draw lines connecting centroids if at least two centroids are found
        if len(centroids) >= 2:
            centroid1, centroid2 = centroids[:2]

            # Draw lines from last centroids to current centroids
            if self.last_centroid1 is not None:
                cv2.line(imdraw, self.last_centroid1, centroid1, (0, 255, 255), 2)
            if self.last_centroid2 is not None:
                cv2.line(imdraw, self.last_centroid2, centroid2, (0, 255, 255), 2)

        #combine the original frame with the drawn lines
        test = cv2.add(test, imdraw)

        return test, centroids, backup_centroids
    
    def run(self, frame):
        #process the frame and obtain the result, current centroids, and backup centroids
        final, centroids, backup_centroids = self.process_frame(frame)
        return final, centroids, backup_centroids
