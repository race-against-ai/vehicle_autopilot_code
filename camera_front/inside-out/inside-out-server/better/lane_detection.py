import time
import cv2
import numpy as np
from vector import Vector
from numpy.typing import NDArray

def time_tracker(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        if not hasattr(self, 'time_data'):
            self.time_data = []
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        self.time_data.append(duration)
        return result
    return wrapper


class LaneDetectionClass:
    def __init__(self):
        
        self.frame : NDArray[np.float64] = None 
        """current frame"""
        self.row_indices_warped : list[int] = [767, 676, 614, 516, 443, 320, 0]
        self.row_indices = [584, 510, 470, 420, 390, 350, 280]
        """search rows"""
        self.blacked_frame : NDArray[np.float64] = None 
        """blacked image: only selected rows are shown"""
        self.binary_frame : NDArray[np.float64] = None 
        """binrary image of the blacked frame"""
        self.roi_frame : NDArray[np.float64] = None
        """region of interest image of the roi method""" 
        self.warped_frame : NDArray[np.float64] = None
        """warped image of the warp method""" 
        self.left_counts : list[int] = None 
        """coords of left line points"""
        self.right_counts : list[int] = None 
        """coords of right line points"""
        self.orig_image_size : list[int] = [1024,768]
    
        self.width : int = self.orig_image_size[0]
        self.height : int = self.orig_image_size[1]

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
        self.padding = int(0.25 * self.width) # padding from side of the image in pixels
        self.desired_roi_points = np.float32([
        [self.padding, 0], # Top-left corner
        [self.padding, self.orig_image_size[1]], # Bottom-left corner         
        [self.orig_image_size[
            0]-self.padding, self.orig_image_size[1]], # Bottom-right corner
        [self.orig_image_size[0]-self.padding, 0] # Top-right corner
        ])

    def set_frame(self, frame):
        self.frame = frame

    def select_rows_black_rest(self):
        """
        Select specific rows of pixels from the frame based on the given row indices and set the rest to black.
        :param frame: The input frame
        :param row_indices: List of row indices to select
        :return: Image with only the selected rows of pixels and the rest blacked out
        """
        selected_rows = self.frame[self.row_indices]
        selected_rows_white = self.white(selected_rows)
        blacked_rest = np.zeros_like(self.frame)
        blacked_rest[self.row_indices] = selected_rows_white
        self.blacked_frame = cv2.cvtColor(blacked_rest, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def white(frame):
        """
        Extract specific white pixels from the input frame.
        :param frame: Input frame (RGB image)
        :return: Image containing only white pixels
        """
        lower_white = np.array([180, 180, 180], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(frame, lower_white, upper_white)
        white_pixels = cv2.bitwise_and(frame, frame, mask=mask)
        return white_pixels
    

    def region_of_interest(self):
        """
        Apply a region of interest mask to the input frame.
        :param frame: Input frame (RGB image)
        :param plot: Boolean indicating whether to display the masked image (default: False)
        :return: Image containing only the region of interest
        """
        # Create a mask of zeros with the same shape as the frame if not created already
        if not hasattr(self, 'mask'):
            self.mask = np.zeros_like(self.blacked_frame)
        # Reset the mask to zeros
        self.mask.fill(0)
        # Define region of interest by filling the area within roi_points with white
        cv2.fillPoly(self.mask, self.warped_points, (255, 255, 255))
        # Bitwise AND between the frame and the mask to keep only the region of interest
        masked_frame = cv2.bitwise_and(self.blacked_frame, self.mask)

        self.roi_frame = masked_frame


    def perspective_transform(self):
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
        self.roi_frame, self.transformation_matrix, self.orig_image_size, flags=(
        cv2.INTER_LINEAR)) 
    
        # Convert image to binary
        (thresh, binary_warped) = cv2.threshold(
        self.warped_frame, 127, 255, cv2.THRESH_BINARY)           
        self.warped_frame = binary_warped

          
    def find_nearest_white_pixels(self):
        """
        Find the nearest white pixels to the middle of the warped frame for each row.
        :param pixel_rows: List of row indices to search for white pixels
        :return: Lists containing the counts of white pixels on the left and right sides for each row
        """
        _, width = self.warped_frame.shape
        middle = width // 2

        self.left_counts = []
        self.right_counts = []

        for row in self.row_indices_warped:
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

            self.left_counts.append(left_count) 
            self.right_counts.append(right_count)


class CurveDetectionClass(LaneDetectionClass):
    def __init__(self):
        super().__init__()
        self.vector = Vector() 
        """class to calculate distance of 3 points"""
        self.straight_counts : list[int] = None 
        """array for cords of straight line"""
        self.curve : bool = None
        """boolean: in curve?"""
        self.brake : bool = None 
        """boolean curve is coming?"""
        self.curve_direction : str = None
        """curve direction"""
        self.curve_validation : ValueCollection = ValueCollection()

        self.rightcounter : int = 0
        """validation counter for right curves"""
        self.straighcounter : int = 0
        """"validation counter for straight lines"""

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

    def is_curve(self):
        """
        Determine if the lane is a curve.
        :param printToTerminal: Boolean indicating whether to print information to the terminal
        :return: Tuple indicating whether it's a curve, and distances for speed and curve detection
        """
        #experimental: if one distance is negative its maybe a straight line:
        if self.negative_numbers():
            return False
        else:
            array = self.left_counts
            if array is not None:
                array = array[:-1]
                #if some points in array are 512 (dashed line f.e.) search for pair of numbers that are not 512, otherwise just use the last
                some_512 = any(count == 512 for count in array)
                if some_512:
                    val = next((val for val in reversed(array) if val != 512), None)
                    position = self.find_position_of_value(array, val)
                    #if position is one of the first two values (4 or 5) it should not use them bcs the line would be straight whatever happens
                    if position == 4 or position == 5:
                        point_to_check = (array[-1],self.row_indices_warped[-2])
                    else:
                        point_to_check = (val,self.row_indices_warped[position])
                else:
                    point_to_check = (array[-1],self.row_indices_warped[-2])
                point1 = (array[0],self.row_indices_warped[0])
                point2 = (array[1],self.row_indices_warped[1])
                distance = self.vector.calculate_distance(point1,point2,point_to_check, debug=False)
                #40 is an assumed value for detecting a curve, might be changed
                if distance < 40:
                    self.curve = False
                else:
                    self.curve = True

    def is_brake(self):
        """
        Determine if the lane is a curve.
        :param printToTerminal: Boolean indicating whether to print information to the terminal
        :return: Tuple indicating whether it's a curve, and distances for speed and curve detection
        """
        array = self.left_counts
        point_to_check = (array[-1],self.row_indices_warped[-1])
        point1 = (array[0],self.row_indices_warped[0])
        point2 = (array[1],self.row_indices_warped[1])
        distance = self.vector.calculate_distance(point1,point2,point_to_check, debug=False)
        #100 is an assumed value for detecting a curve, might be changed
        if distance < 100:
            self.brake = False
        else:
            self.brake = True
            
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



    def left_or_right(self):
        """
        Determine whether the vehicle is moving left or right.

        This method analyzes the current state of the vehicle to determine its direction of movement. It considers various
        factors such as whether the vehicle is in a curve, if braking is active, and if the line has crossed zero.
        :param curve and brake (see is_curve and is_brake)
        :return: string if curve is turning "left" or "right"
        """

        if self.left_counts[0] != 513 and self.left_counts[2] != 513 and self.left_counts[-1] != 513:

            if self.curve == False and self.brake == True:
                if None == self.curve_validation.most_common_value():
                    self.curve_validation.reset()

                point_to_check = (512 - self.left_counts[-1],self.row_indices_warped[-1])

                point1 = (512 - self.left_counts[0],self.row_indices_warped[0])
                point2 = (512 - self.left_counts[2],self.row_indices_warped[2])

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
                    self.straighcounter = 0


            if self.curve == False and self.brake == False:

                if self.straighcounter > 10:
                    self.rightcounter = 0
                    self.straighcounter = 0

                else:
                    self.straighcounter +=1

                self.curve_validation.add_value(None)

                if None == self.curve_validation.most_common_value():

                    self.curve_validation.reset()
                    self.curve_direction = None


    def validate_straight(self):
        pass

    def validate_right(self):
        pass

class SteeringDecidingClass(CurveDetectionClass):
    def __init__(self):
        super().__init__()
        self.left_offset : float = None
        """cars left offset (in percentage)"""
        self.right_offset : float = None
        """cars left offset (in percentage)"""
        self.offset = 0


    def steering_decision(self):
        self.steer_for_right()
        # if self.curve_direction == None:
        #     #left curve
        #     self.steer_for_left()
        # elif self.curve_direction == "right":
        #     #right curve
        #     self.steer_for_right()


    def steer_for_right(self):
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

        curve = True

        if left_all_512:
            distance_left = 512
        elif left_some_512:
            if curve:
                distance_left = next((val for val in self.left_counts[1:-2] if val != 512), None)
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
                distance_left = next((val for val in self.left_counts[1:-2] if val != 512), None)
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
            
        self.left_offset = distance_left
        self.right_offset = distance_right

    def steer_for_left(self):
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
            if self.curve:
                distance_left = next((val for val in self.left_counts if val != 512), None)
            else:
                distance_left = next((val for val in self.left_counts if val != 512), None)
            self.dashed_left = True
        elif left_no_512:
            if self.curve:
                distance_left = next((val for val in self.left_counts if val != 512), None)
            else:
                distance_left = next((val for val in self.left_counts if val != 512), None)
            self.dashed_left = False

        if right_all_512:
            distance_right = 512
        elif right_some_512:
            if self.curve:
                distance_right = next((val for val in self.right_counts[1:-2] if val != 512), None)
                if distance_right == None or distance_right == 0:
                    if self.right_counts[0] != 512 or self.right_counts[0] != 0:
                        distance_right = self.right_counts[0]
                    else:
                        distance_right = 512
            else:
                distance_right = next((val for val in self.right_counts if val != 512), None)
            self.dashed_right = True
        elif right_no_512:
            if self.curve:
                distance_right = next((val for val in self.right_counts[1:-2] if val != 512), None)
            else:
                distance_right = next((val for val in self.right_counts if val != 512), None)
            self.dashed_right = False

        self.left_offset = distance_left
        self.right_offset = distance_right
    

    def calculate_steering_angle(self):
        
        self.offset = self.left_offset - self.right_offset

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
        print("list cleared")


class TimerClass(SteeringDecidingClass):
    def __init__(self):
        super().__init__()
        self.time_data : list[float] = []

    @time_tracker
    def select_rows_black_rest(self):
        return super().select_rows_black_rest()

    @time_tracker
    def perspective_transform(self):
        return super().perspective_transform()

    @time_tracker
    def find_nearest_white_pixels(self):
        return super().find_nearest_white_pixels()

    @time_tracker
    def is_curve(self):
        return super().is_curve()

    @time_tracker
    def is_brake(self):
        return super().is_brake()

    @time_tracker
    def steering_decision(self):
        return super().steering_decision()

def call_lane_detection(LaneDetectorInstance, frame):
    """
    Call the lane detection process using the provided LaneDetectorInstance and frame.
    This function sequentially calls methods of the LaneDetectorInstance to perform lane detection on the given frame.
    """
    LaneDetectorInstance.set_frame(frame)

    LaneDetectorInstance.select_rows_black_rest()
    LaneDetectorInstance.region_of_interest()
    LaneDetectorInstance.perspective_transform()
    LaneDetectorInstance.find_nearest_white_pixels()
    LaneDetectorInstance.is_curve()
    LaneDetectorInstance.is_brake()
    LaneDetectorInstance.left_or_right()
    LaneDetectorInstance.steering_decision()
    LaneDetectorInstance.calculate_steering_angle()

    data = LaneDetectorInstance.time_data
    LaneDetectorInstance.time_data = []
    return LaneDetectorInstance.offset, data, LaneDetectorInstance.curve, LaneDetectorInstance.brake