import cv2
import numpy as np
import time
import matplotlib.pyplot as plt # Used for plotting and error checking

class LaneDetection:
    def __init__(self, orig_frame):
        #original frame
        self.orig_frame = orig_frame

        # (Width, Height) of the original video frame (or image)
        self.orig_image_size = self.orig_frame.shape[::-1][1:]
    
        width = self.orig_image_size[0]
        height = self.orig_image_size[1]
        self.width = width
        self.height = height
            

        #position of the car in the image
        self.car_cords = np.array([[100, 800], [900, 800], [500, 400]], dtype=np.int32)

        self.warped_points = np.array([[(380, 280), (700, 280), (1044, 585), (-20, 585)]], dtype=np.int32)

        self.roi_points = np.float32([
            (380,280), # Top-left corner
            (-20, 585), # Bottom-left corner            
            (1064,585), # Bottom-right corner
            (720,280) # Top-right corner
        ])

        # The desired corner locations  of the region of interest
        # after we perform perspective transformation.
        # Assume image width of 1024, padding == 150.
        self.padding = int(0.25 * width) # padding from side of the image in pixels
        self.desired_roi_points = np.float32([
        [self.padding, 0], # Top-left corner
        [self.padding, self.orig_image_size[1]], # Bottom-left corner         
        [self.orig_image_size[
            0]-self.padding, self.orig_image_size[1]], # Bottom-right corner
        [self.orig_image_size[0]-self.padding, 0] # Top-right corner
        ]) 

        # This will hold the image after perspective transformation
        self.warped_frame = None
        self.transformation_matrix = None
        self.inv_transformation_matrix = None

        # Histogram that shows the white pixel peaks for lane line detection
        self.histogram = None

        #distances to the lines
        self.left_offset = None
        self.right_offset = None
            
        # Pixel parameters for x and y dimensions
        self.YM_PER_PIX = 4.0 / 768 # meters per pixel in y dimension
        self.XM_PER_PIX = 2.0 / 1024 # meters per pixel in x dimension
            
        # Radii of curvature and offset
        self.center_offset = None

        #offset of the car
        self.car_offset = 27

        #arrays for detected line points
        self.left_counts = None
        self.right_counts = None

        #dashed variables
        self.dashed_left = False
        self.dashed_right = False
        self.switch_to = ""

    def find_first_white_pixel(self, image):
        """
        function to find pixels in warped image (use it if u change the row indeces)
        """
        white_pixels = []
        for row_index, row in enumerate(image):
            # Find the index of the first white pixel in the row
            white_pixel_index = np.where(row == 255)[0]
            if len(white_pixel_index) > 0:
                white_pixel_index = white_pixel_index[0]
                white_pixels.append((row_index, white_pixel_index))
        print(white_pixels)

    def select_rows_black_rest(self, frame, row_indices):
        """
        Select specific rows of pixels from the frame based on the given row indices and set the rest to black.
        :param frame: The input frame
        :param row_indices: List of row indices to select
        :return: Image with only the selected rows of pixels and the rest blacked out
        """
        selected_rows = frame[row_indices]

        selected_rows_white = self.white(selected_rows)

        blacked_rest = np.zeros_like(frame)
        blacked_rest[row_indices] = selected_rows_white

        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(blacked_rest, cv2.COLOR_RGB2GRAY)

        return gray

    def white(self, frame):
        """
        Detect white regions in the given frame.
        :param frame: The input frame
        :return: Image with white regions detected
        """

        # Define range of white color in RGB
        lower_white = np.array([180, 180, 180], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)

        # Threshold the RGB image to get only white colors
        mask = cv2.inRange(frame, lower_white, upper_white)

        # Bitwise-AND mask and original image
        white_pixels = cv2.bitwise_and(frame, frame, mask=mask)

        return white_pixels


    def region_of_interest(self, frame, plot=False):
        """
        Mask the region of interest (ROI) on the given frame.
        :param frame: The input frame to mask the ROI on
        :param plot: Whether to display the plotted image or not
        """
        # Create a mask of zeros with the same shape as the frame
        mask = np.zeros_like(frame)

        # Define region of interest by filling the area within roi_points with white
        cv2.fillPoly(mask, self.warped_points, (255, 255, 255))

        # Bitwise AND between the frame and the mask to keep only the region of interest
        masked_frame = cv2.bitwise_and(frame, mask)

        if plot:
            cv2.imshow('Masked Image', masked_frame)

        return masked_frame

    
    def perspective_transform(self, frame=None, plot=False):
        """
        Perform the perspective transform.
        :param: frame Current frame
        :param: plot Plot the warped image if True
        :return: Bird's eye view of the current lane
        """
                
        # Calculate the transformation matrix
        self.transformation_matrix = cv2.getPerspectiveTransform(
        self.roi_points, self.desired_roi_points)
    
        # Calculate the inverse transformation matrix           
        self.inv_transformation_matrix = cv2.getPerspectiveTransform(
        self.desired_roi_points, self.roi_points)
    
        # Perform the transform using the transformation matrix
        self.warped_frame = cv2.warpPerspective(
        frame, self.transformation_matrix, self.orig_image_size, flags=(
        cv2.INTER_LINEAR)) 
    
        # Convert image to binary
        (thresh, binary_warped) = cv2.threshold(
        self.warped_frame, 127, 255, cv2.THRESH_BINARY)           
        self.warped_frame = binary_warped
    
        # Display the perspective transformed (i.e. warped) frame
        if plot == True:
            warped_copy = self.warped_frame.copy()
            warped_plot = cv2.polylines(warped_copy, np.int32([
                            self.desired_roi_points]), True, (147,20,255), 3)
            
            cv2.imshow('Warped Image with ROI', warped_plot)
    
 
        return self.warped_frame
    
        
    def find_nearest_white_pixels(self, pixel_rows):
        height, width = self.warped_frame.shape
        middle = width // 2

        left_counts = []
        right_counts = []

        for row in pixel_rows:
            left_count = 0
            for i in range(middle, -1, -1):
                if self.warped_frame[row, i] == 255:
                    break
                left_count += 1

            right_count = 0
            for i in range(middle, width):
                if self.warped_frame[row, i] == 255:
                    break
                right_count += 1

            left_counts.append(left_count)
            right_counts.append(right_count)

        self.left_counts = left_counts
        self.right_counts = right_counts

        return left_counts, right_counts
    
    def dashed_side(self, printToTerminal):

        #just for straight
        #if curve == False

        distance_left = None
        distance_right = None

        info_left = None
        info_right = None

        left_all_512 = all(count == 512 for count in self.left_counts)
        right_all_512 = all(count == 512 for count in self.right_counts)

        left_some_512 = any(count == 512 for count in self.left_counts)
        right_some_512 = any(count == 512 for count in self.right_counts)

        left_no_512 = not any(count == 512 for count in self.left_counts)
        right_no_512 = not any(count == 512 for count in self.right_counts)

        #left all 512 = no line left
        if left_all_512:
            if printToTerminal:
                info_left =  "No line on the left"
            distance_left = 512
        #left some 512 = dashed on the left -> use first
        elif left_some_512:
            if printToTerminal:
                info_left =  "Dashed line on the left"
            distance_left = next((val for val in self.left_counts if val != 512), None)
            self.dashed_left = True
        #left no 512 = straight line -> use first
        elif left_no_512:
            if printToTerminal:
                info_left = "Straight line on the left"
            distance_left = next((val for val in self.left_counts if val != 512), None)
            self.dashed_left = False
        
        #right all 512 = no line right
        if right_all_512:
            if printToTerminal:
                info_right =  "No line on the right"
            distance_right = 512
        #right some 512 = dashed on the right -> use first
        elif right_some_512:
            if printToTerminal:
                info_right =  "Dashed line on the right"
            distance_right = next((val for val in self.right_counts if val != 512), None)
            self.dashed_right = True
        #right no 512 = straight line -> use first
        elif right_no_512:
            if printToTerminal:
                info_right = "Straight line on the right"
            distance_right = next((val for val in self.right_counts if val != 512), None)
            self.dashed_right = False

        if printToTerminal:
            print(info_left)
            print(f'Distance left: {distance_left}')
            print(info_right)
            print(f'Distance right: {distance_right}')

        self.left_offset = distance_left
        self.right_offset = distance_right

        return distance_left, distance_right

        #if curve == True

    def switch_lane(self, direction = ""):

        #function for switchting the lane

        #get lane by dashed_right or dashed_left

        self.switch_to = direction if direction in ["left", "right"] else ""

        #switch to left             and dashed on left then car is on the right
        if self.switch_to == "left" and self.dashed_left == True:
            #switch to left -> dashed right has to be true
            pass
        #car is already on the left
        elif self.switch_to == "left" and self.dashed_left == False:
            self.switch_to = ""


        #switch to right             and dashed on right then car is on the left
        if self.switch_to == "right" and self.dashed_right == True:
            #switch to right -> dashed left has to be true
            pass
        #car is already on the right
        if self.switch_to == "right" and self.dashed_right == False:
            self.switch_to = ""


    def is_curve(self, printToTerminal=False):

        if self.left_counts is not None:
            first_point = self.left_counts[0]
            second_point = self.left_counts[-1]

            if 0 < (first_point - second_point) < 100:
                if printToTerminal:
                    print(f'Offset einer Linie für Kurve {first_point - second_point}')
                    print('Gerade')
                return False
            else:
                if printToTerminal:
                    print(f'Offset einer Linie für Kurven {first_point - second_point}')
                    print('Kurve')
                return True


    def calculate_steering_angle(self):
        
        offset = self.left_offset - self.right_offset

        return offset
    
def main_lanes(frame, lane_detection, debug):
    """
    Perform main lane detection processes including white detection, blending out cars, region of interest (ROI) extraction,
    perspective transformation, histogram calculation, lane line detection using sliding windows, filling in the lane lines,
    overlaying lines on the original frame, and calculating the car's position offset.
    :param frame: The input frame for lane detection
    :return: Tuple containing the car's center offset and time data collected during the process
    """
     # Row indices to select
    row_indices = [584, 480, 430, 390, 360, 300]
    #row_indices = [767, 690, 600, 520, 400, 250]

    roi_start_time = time.time()
    # Select specific rows of pixels and set the rest to black
    blacked_image = lane_detection.select_rows_black_rest(frame, row_indices)
    roi_time = (time.time() - roi_start_time) * 1000
    
    #show region of interest by displaying them
    #roi_start_time = time.time()
    #roi_image = lane_detection.region_of_interest(blacked_image, plot=False)
    #roi_time = (time.time() - roi_start_time) * 1000

    #show only white pixels in image
    #white_start_time = time.time()
    #white_image = lane_detection.white(blacked_image)
    #white_time = (time.time() - white_start_time) * 1000
    white_time = 0

    #transform perspective
    cropped_start_time = time.time()
    cropped_image = lane_detection.perspective_transform(blacked_image, False)
    transform_time = (time.time() - cropped_start_time) * 1000

    #lane_detection.find_first_white_pixel(cropped_image)


    # Find lane line pixels using the sliding window method 
    find_start_time = time.time()
    left_counts, right_counts = lane_detection.find_nearest_white_pixels([767, 660, 580, 494, 408, 140])
    find_time = (time.time() - find_start_time) * 1000

    #check for dashed side so distance is calculatet right
    dashed_start_time = time.time()
    distance_left, distance_right = lane_detection.dashed_side(debug)
    dashed_time = (time.time() - dashed_start_time) * 1000

    #calculate the offset
    center_offset = lane_detection.calculate_steering_angle()

    #function for switching lane (TO_DO)
    #lane_detection.switch_lane("")

    #function to detect curve
    straight = lane_detection.is_curve(debug)

    data = [time.strftime("%H:%M:%S"),
        white_time,
        roi_time,
        transform_time,
        find_time,
        dashed_time]

    return center_offset, data, straight
