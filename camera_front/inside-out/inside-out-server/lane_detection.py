import cv2
import numpy as np
import time
from vector import Vector

class LaneDetection:
    def __init__(self, orig_frame):
        #original frame
        self.orig_frame = orig_frame

        # (Width, Height) of the original video frame (or image)
        self.orig_image_size = [1024,768]
    
        width = self.orig_image_size[0]
        height = self.orig_image_size[1]
        self.width = width
        self.height = height
            

        #position of the car in the image
        self.car_cords = np.array([[100, 800], [900, 800], [500, 400]], dtype=np.int32)

        self.warped_points = np.array([[(300, 280), (724, 280), (1024, 585), (0, 585)]], dtype=np.int32)

        self.roi_points = np.float32([
            (300,280), # Top-left corner
            (0, 585), # Bottom-left corner            
            (1024,585), # Bottom-right corner
            (724,280) # Top-right corner
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
        self.car_offset = 20

        #arrays for detected line points
        self.left_counts = None
        self.right_counts = None

        #dashed variables
        self.dashed_left = False
        self.dashed_right = False
        self.switch_to = ""

        #class to calculate distance of 3 points
        self.vector = Vector()

        self.rows_to_search = [767, 676, 614, 516, 443, 320, 0]

        self.curve_validation = ValueCollection()

        self.curve_direction = None

        self.rightcounter = 0
        self.straighcounter = 0

    def find_first_white_pixel(self, image):
        """
        Find the coordinates of the first white pixel for every row in the image.
        :param image: Input image as a NumPy array
        :print: pixel-coordinates
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
        Extract specific white pixels from the input frame.
        :param frame: Input frame (RGB image)
        :return: Image containing only white pixels
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
        Apply a region of interest mask to the input frame.
        :param frame: Input frame (RGB image)
        :param plot: Boolean indicating whether to display the masked image (default: False)
        :return: Image containing only the region of interest
        """
        # Create a mask of zeros with the same shape as the frame if not created already
        if not hasattr(self, 'mask'):
            self.mask = np.zeros_like(frame)

        # Reset the mask to zeros
        self.mask.fill(0)

        # Define region of interest by filling the area within roi_points with white
        cv2.fillPoly(self.mask, self.warped_points, (255, 255, 255))

        # Bitwise AND between the frame and the mask to keep only the region of interest
        masked_frame = cv2.bitwise_and(frame, self.mask)

        if plot:
            cv2.imshow('Masked Image', masked_frame)

        return masked_frame


    
    def perspective_transform(self, frame=None, plot=False):
        """
        Perform the perspective transform.
        :param: frame: current frame
        :param: plot: Plot the warped image if True
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

        #cv2.imwrite("test.png", self.warped_frame)
    
        # Display the perspective transformed (i.e. warped) frame
        if plot == True:
            warped_copy = self.warped_frame.copy()
            warped_plot = cv2.polylines(warped_copy, np.int32([
                            self.desired_roi_points]), True, (147,20,255), 3)
            
            cv2.imshow('Warped Image with ROI', warped_plot)
    
 
        return self.warped_frame
    
        
    def find_nearest_white_pixels(self, pixel_rows):
        """
        Find the nearest white pixels to the middle of the warped frame for each row.
        :param pixel_rows: List of row indices to search for white pixels
        :return: Lists containing the counts of white pixels on the left and right sides for each row
        """
        height, width = self.warped_frame.shape
        middle = width // 2

        left_counts = []
        right_counts = []

        for row in pixel_rows:
            left_count = 0
            for i in range(middle, -1, -1):
                if self.warped_frame.item(row, i) == 255:
                    break
                left_count += 1

            right_count = 0
            for i in range(middle, width):
                if self.warped_frame.item(row, i) == 255:
                    break
                right_count += 1

            left_counts.append(left_count) 
            right_counts.append(right_count)

        self.left_counts = left_counts
        self.right_counts = right_counts

        return left_counts, right_counts
    

    def steering_decision(self, curve):
        if self.curve_direction == None:
            #left curve
            self.steer_for_left(curve)
        else:
            #right curve
            self.steer_for_right(curve)


    def steer_for_right(self, curve):
        """
        Determine the distance and type of dashed lines on the left and right sides.
        :param printToTerminal: Boolean indicating whether to print information to the terminal
        :param curve: Boolean indicating whether the current lane is a curve or straight
        :return: Tuple containing distance and information about dashed lines on the left and right sides
        """
        distance_left = None
        distance_right = None

        left_all_512 = all(count == 512 for count in self.left_counts)
        right_all_512 = all(count == 512 for count in self.right_counts)
        left_some_512 = any(count == 512 for count in self.left_counts)
        right_some_512 = any(count == 512 for count in self.right_counts)
        left_no_512 = not any(count == 512 for count in self.left_counts)
        right_no_512 = not any(count == 512 for count in self.right_counts)

        #curve = True

        if left_all_512:
            distance_left = 512
        elif left_some_512:
            if curve:
                distance_left = next((val for val in self.left_counts[3:-2] if val != 512), None)
                if distance_left == None or distance_left == 0:
                    if self.left_counts[0] != 512 or self.left_counts[0] != 0:
                        distance_left = self.left_counts[0]
                    else:
                        distance_left = 512
            else:
                distance_left = next((val for val in self.left_counts if val != 512), None)
            self.dashed_left = True
        elif left_no_512:
            if curve:
                distance_left = next((val for val in self.left_counts[3:-2] if val != 512), None)
            else:
                distance_left = next((val for val in self.left_counts if val != 512), None)
            self.dashed_left = False

        if right_all_512:
            distance_right = 512
        elif right_some_512:
            if curve:
                distance_right = next((val for val in self.right_counts if val != 512), None)
            else:
                distance_right = next((val for val in self.right_counts if val != 512), None)
            self.dashed_right = True
        elif right_no_512:
            if curve:
                distance_right = next((val for val in self.right_counts if val != 512), None)
            else:
                distance_right = next((val for val in self.right_counts if val != 512), None)
            self.dashed_right = False
            
        self.left_offset = distance_left - 60
        self.right_offset = distance_right

    def steer_for_left(self, curve):
        """
        Determine the distance and type of dashed lines on the left and right sides.
        :param printToTerminal: Boolean indicating whether to print information to the terminal
        :param curve: Boolean indicating whether the current lane is a curve or straight
        :return: Tuple containing distance and information about dashed lines on the left and right sides
        """
        distance_left = None
        distance_right = None

        left_all_512 = all(count == 512 for count in self.left_counts)
        right_all_512 = all(count == 512 for count in self.right_counts)
        left_some_512 = any(count == 512 for count in self.left_counts)
        right_some_512 = any(count == 512 for count in self.right_counts)
        left_no_512 = not any(count == 512 for count in self.left_counts)
        right_no_512 = not any(count == 512 for count in self.right_counts)

        if left_all_512:
            distance_left = 512
        elif left_some_512:
            if curve:
                distance_left = next((val for val in self.left_counts if val != 512), None)
            else:
                distance_left = next((val for val in self.left_counts if val != 512), None)
            self.dashed_left = True
        elif left_no_512:
            if curve:
                distance_left = next((val for val in self.left_counts if val != 512), None)
            else:
                distance_left = next((val for val in self.left_counts if val != 512), None)
            self.dashed_left = False

        if right_all_512:
            distance_right = 512
        elif right_some_512:
            if curve:
                distance_right = next((val for val in self.right_counts[2:-2] if val != 512), None)
                if distance_right == None or distance_right == 0:
                    if self.right_counts[0] != 512 or self.right_counts[0] != 0:
                        distance_right = self.right_counts[0]
                    else:
                        distance_right = 512
            else:
                distance_right = next((val for val in self.right_counts if val != 512), None)
            self.dashed_right = True
        elif right_no_512:
            if curve:
                distance_right = next((val for val in self.right_counts[2:-2] if val != 512), None)
            else:
                distance_right = next((val for val in self.right_counts if val != 512), None)
            self.dashed_right = False


        self.left_offset = distance_left
        self.right_offset = distance_right


    def switch_lane(self, direction = ""):
        """
        Function for switching lanes.
        :param curve: Boolean indicating whether the current lane is a curve or straight
        :param direction: Direction to switch to ("left" or "right")
        :return: fake offset for lane switching
        """
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

    def negative_numbers(self):
        """
        Check if any of the values in the left_counts or right_counts lists are negative.
        :return: Boolean indicating whether any negative numbers are found
        """
        for num1, num2 in zip(self.left_counts, self.right_counts):

            if num1 < 0:
                return True
            if num2 < 0:
                return True
        
        return False
    
    @staticmethod
    def find_position_of_value(array, value_to_find):
        """
        Find the position of a value within an array.
        :param array: Input array to search
        :param value_to_find: Value to find within the array
        :return: Position of the value within the array
        """
        for i, val in enumerate(reversed(array)):
            if val != value_to_find:
                return len(array) - i - 1  # Calculate the position of the value

    def is_curve(self, printToTerminal=False):
        """
        Determine if the lane is a curve.
        :param printToTerminal: Boolean indicating whether to print information to the terminal
        :return: Tuple indicating whether it's a curve, and distances for speed and curve detection
        """
        #experimental: if one distance is negative its maybe a straight line:
        if self.negative_numbers():

            return False
        
        else:

            if self.dashed_left:
                array = self.right_counts
            elif self.dashed_right:
                array = self.left_counts
            else:
                array = self.left_counts

            if array is not None:

                array = array[:-1]

                #if some points in array are 512 (dashed line f.e.) search for pair of numbers that are not 512, otherwise just use the last
                some_512 = any(count == 513 for count in array)
                if some_512:
                    val = next((val for val in reversed(array) if val != 513), None)
                    position = self.find_position_of_value(array, val)
                    #if position is one of the first two values (4 or 5) it should not use them bcs the line would be straight whatever happens
                    if position == 4 or position == 5:
                        point_to_check = (array[-1],self.rows_to_search[-2])
                    else:
                        point_to_check = (val,self.rows_to_search[position])
                else:
                    point_to_check = (array[-1],self.rows_to_search[-2])

                point1 = (array[0],self.rows_to_search[0])
                point2 = (array[1],self.rows_to_search[1])
                distance = self.vector.calculate_distance(point1,point2,point_to_check, debug=False)

                #40 is an assumed value for detecting a curve, might be changed
                if distance < 40:
                    if printToTerminal:
                        print(f'Offset einer Linie für Kurve {distance}')
                        print('Gerade')
                    return False, distance
                else:
                    if printToTerminal:
                        print(f'Offset einer Linie für Kurven {distance}')
                        print('Kurve')
                    return True, distance
                
    def is_brake(self, printToTerminal=False):

        array = self.left_counts

        point_to_check = (array[-1],self.rows_to_search[-1])

        point1 = (array[0],self.rows_to_search[0])
        point2 = (array[1],self.rows_to_search[1])
        distance = self.vector.calculate_distance(point1,point2,point_to_check, debug=False)

        #40 is an assumed value for detecting a curve, might be changed
        if distance < 100:
            if printToTerminal:
                print(f'Offset einer Linie für Kurve {distance}')
                print('Gerade')
            return False, distance
        else:
            if printToTerminal:
                print(f'Offset einer Linie für Kurven {distance}')
                print('Kurve')
            return True, distance


    def calculate_steering_angle(self):
        
        offset = self.left_offset - self.right_offset - self.car_offset

        return offset
    

    
    def left_or_right(self, curve, brake):
        """
        Determine whether the vehicle is moving left or right.

        This method analyzes the current state of the vehicle to determine its direction of movement. It considers various
        factors such as whether the vehicle is in a curve, if braking is active, and if the line has crossed zero.
        :param curve and brake (see is_curve and is_brake)
        :return: string if curve is turning "left" or "right"
        """

        if self.left_counts[0] != 513 and self.left_counts[2] != 513 and self.left_counts[-1] != 513:

            if curve == False and brake == True:
                if None == self.curve_validation.most_common_value():
                    self.curve_validation.reset()

                point_to_check = (512 - self.left_counts[-1],self.rows_to_search[-1])

                point1 = (512 - self.left_counts[0],self.rows_to_search[0])
                point2 = (512 - self.left_counts[2],self.rows_to_search[2])

                self.curve_direction = self.vector.calculate_distance(point1,point2,point_to_check, True, debug=False)

                # if self.curve_direction == "right" and self.curve_validation.most_common_value() != "right":
                #     self.curve_validation.reset()
                #     self.curve_direction = "left"
                #     self.curve_validation.add_value("right")

                # else:

                self.curve_validation.add_value(self.curve_direction)

                self.curve_direction = self.curve_validation.most_common_value()

                if self.curve_direction == "right":
                    self.rightcounter += 1

                if self.rightcounter > 10:
                    self.curve_direction = "right"


            if curve == False and brake == False:

                if self.straighcounter > 10:
                    self.rightcounter = 0
                    self.straighcounter = 0
                else:
                    self.straighcounter +=1

                self.curve_validation.add_value(None)

                if None == self.curve_validation.most_common_value():

                    self.curve_validation.reset()
                    self.curve_direction = None

    
def main_lanes(frame, lane_detection, debug):
    """
    Perform main lane detection processes including white detection, blending out cars, region of interest (ROI) extraction,
    perspective transformation, histogram calculation, lane line detection using sliding windows, filling in the lane lines,
    overlaying lines on the original frame, and calculating the car's position offset.
    :param frame: The input frame for lane detection
    :return: Tuple containing the car's center offset and time data collected during the process
    """
     # Row indices to select
    row_indices = [584, 510, 470, 420, 390, 350, 280] #290

    roi_start_time = time.time()
    # Select specific rows of pixels and set the rest to black
    blacked_image = lane_detection.select_rows_black_rest(frame, row_indices)
    roi = lane_detection.region_of_interest(blacked_image)
    roi_time = (time.time() - roi_start_time) * 1000

    #transform perspective
    cropped_start_time = time.time()
    cropped_image = lane_detection.perspective_transform(roi, False)
    transform_time = (time.time() - cropped_start_time) * 1000

    #lane_detection.find_first_white_pixel(cropped_image)

    # Find lane line pixels using the sliding window method 
    find_start_time = time.time()
    left_counts, right_counts = lane_detection.find_nearest_white_pixels([767, 676, 614, 516, 443, 320, 0]) #69
    find_time = (time.time() - find_start_time) * 1000

    #function to detect curve
    curve, distance = lane_detection.is_curve(debug)
    speed, distance_speed = lane_detection.is_brake(debug)


    lane_detection.left_or_right(curve, speed)

    #check for dashed side so distance is calculatet right
    dashed_start_time = time.time()
    lane_detection.steering_decision(curve)
    dashed_time = (time.time() - dashed_start_time) * 1000

    #calculate the offset
    center_offset = lane_detection.calculate_steering_angle()

    #function for switching lane (TO_DO)
    #lane_detection.switch_lane("")

    data = [
        roi_time,
        transform_time,
        find_time,
        dashed_time]

    #save_number_to_file(curve, distance_left, distance_right, distance)

    return center_offset, data, curve, speed, cropped_image

def save_number_to_file(curve, distance_left, distance_right, distance):
    filename = 'curve.txt'
    with open(filename, 'a') as file:  # Use 'a' mode to append to the file
        file.write(f"{curve}, {distance_left}, {distance_right}, {distance}\n")



class ValueCollection:
    def __init__(self):
        self.values = []

    def add_value(self, value):
        """Add a value to the collection."""
        self.values.append(value)

    def most_common_value(self):
        """Return the most common value in the collection.
        If multiple values occur with the same frequency, the first one encountered is returned."""
        if not self.values:
            return None
        return max(set(self.values), key=self.values.count)

    def reset(self):
        """Reset the collection (clear all values)."""
        self.values.clear()
