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

        self.roi_points = np.float32([
            (380,280), # Top-left corner
            (-20, 585), # Bottom-left corner            
            (1044,585), # Bottom-right corner
            (700,280) # Top-right corner
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

        # Sliding window parameters
        self.no_of_windows = 10
        self.margin = int((1/12) * width)  # Window width is +/- margin
        self.minpix = int((1/24) * width)  # Min no. of pixels to recenter window
            
        # Best fit polynomial lines for left line and right line of the lane
        self.left_fit = None
        self.right_fit = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None
        self.leftx = None
        self.rightx = None
        self.lefty = None
        self.righty = None
            
        # Pixel parameters for x and y dimensions
        self.YM_PER_PIX = 4.0 / 768 # meters per pixel in y dimension
        self.XM_PER_PIX = 2.0 / 1024 # meters per pixel in x dimension
            
        # Radii of curvature and offset
        self.center_offset = None

        #offset of the car
        self.car_offset = 27


    def white(self, frame):
        """
        Detect white regions in the given frame.
        :param frame: The input frame
        :return: Image with white regions detected
        """

        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Define range of white color in HSV
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([100, 100, 255])

        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Bitwise-AND mask and original image
        white_pixels = cv2.bitwise_and(frame, frame, mask=mask)

        gray = cv2.cvtColor(white_pixels, cv2.COLOR_HSV2RGB)

        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        return blur
    
    def blend_out_car(self, frame):
        """
        Blend out detected cars from the given frame.
        :param frame: The input frame with detected cars
        :return: Frame with cars blended out
        """
        # Reshape the array into a 3D array with one row
        pts = self.car_cords.reshape((-1, 1, 2))

        # Draw the filled triangle on the black image
        cv2.fillPoly(frame, [pts], color=(0, 0, 0))
    
        return frame

    def plot_region_of_interest(self, frame, plot = False):
        """
        Plot the region of interest (ROI) on the given frame.
        :param frame: The input frame to plot the ROI on
        :param plot: Whether to display the plotted image or not
        """
        # Define region of interest
        # Overlay trapezoid on the frame
        if plot:
            masked = cv2.polylines(frame, np.int32([self.roi_points]), True, (147, 20, 255), 3)
            
            cv2.imshow('Plot Image', masked)
    
    
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
    
    def calculate_histogram(self,frame=None,plot=True):
        """
        Calculate the image histogram to find peaks in white pixel count
            
        :param frame: The warped image
        :param plot: Create a plot if True
        """
                
        # Generate the histogram
        self.histogram = np.sum(frame[int(
                frame.shape[0]/2):,:], axis=0)
    
        if plot == True:
            # Draw both the image and the histogram
            figure, (ax1, ax2) = plt.subplots(2,1) # 2 row, 1 columns
            figure.set_size_inches(10, 5)
            ax1.imshow(frame, cmap='gray')
            ax1.set_title("Warped Binary Frame")
            ax2.plot(self.histogram)
            ax2.set_title("Histogram Peaks")
            plt.show()
                
        return self.histogram
    
    def get_lane_line_indices_sliding_windows(self, plot=False):
        """
        Get the indices of the lane line pixels using the sliding windows technique.
            
        :param: plot Show plot or not
        :return: Best fit lines for the left and right lines of the current lane 
        """
        # Sliding window width is +/- margin
        margin = self.margin

        frame_sliding_window = self.warped_frame.copy()

        # Set the height of the sliding windows
        window_height = int(self.warped_frame.shape[0]/self.no_of_windows)

        # Find the x and y coordinates of all the nonzero 
        # (i.e. white) pixels in the frame. 
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1]) 

        # Store the pixel indices for the left and right lane lines
        left_lane_inds = []
        right_lane_inds = []

        # Current positions for pixel indices for each window,
        # which we will continue to update
        leftx_base, rightx_base = self.histogram_peak()
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Go through one window at a time
        no_of_windows = self.no_of_windows

        for window in range(no_of_windows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
            win_y_high = self.warped_frame.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(frame_sliding_window,(win_xleft_low,win_y_low),(
                win_xleft_high,win_y_high), (255,255,255), 2)
            cv2.rectangle(frame_sliding_window,(win_xright_low,win_y_low),(
                win_xright_high,win_y_high), (255,255,255), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xleft_low) & (
                                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xright_low) & (
                                    nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on mean position
            minpix = self.minpix
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        try:
            # Extract the pixel coordinates for the left and right lane lines
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds] 
            righty = nonzeroy[right_lane_inds]

            # Fit a second order polynomial curve to the pixel coordinates for
            # the left and right lane lines
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2) 

            self.left_fit = left_fit
            self.right_fit = right_fit

            if plot==True:

                # Create the x and y values to plot on the image  
                ploty = np.linspace(
                    0, frame_sliding_window.shape[0]-1, frame_sliding_window.shape[0])
                left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

                # Generate an image to visualize the result
                out_img = np.dstack((
                    frame_sliding_window, frame_sliding_window, (
                    frame_sliding_window))) * 255

                # Add color to the left line pixels and right line pixels
                out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
                out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                    0, 0, 255]

                # Plot the figure with the sliding windows
                figure, (ax1, ax2, ax3) = plt.subplots(3,1) # 3 rows, 1 column
                figure.set_size_inches(10, 10)
                figure.tight_layout(pad=3.0)
                ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
                ax2.imshow(frame_sliding_window, cmap='gray')
                ax3.imshow(out_img)
                ax3.plot(left_fitx, ploty, color='yellow')
                ax3.plot(right_fitx, ploty, color='yellow')
                print(left_fitx)
                print(ploty)
                ax1.set_title("Original Frame")  
                ax2.set_title("Warped Frame with Sliding Windows")
                ax3.set_title("Detected Lane Lines with Sliding Windows")
                plt.show()

            return self.left_fit, self.right_fit

        except TypeError:
            print("Could not fit polynomial due to empty vector for x")
            return None, None

    
    def get_lane_line_previous_window(self, left_fit, right_fit, plot=False):
        """
        Use the lane line from the previous sliding window to get the parameters
        for the polynomial line for filling in the lane line
        :param: left_fit Polynomial function of the left lane line
        :param: right_fit Polynomial function of the right lane line
        :param: plot To display an image or not
        """
        # margin is a sliding window parameter
        margin = self.margin

        # Check if left_fit and right_fit are None
        if left_fit is None or right_fit is None:
            print("Could not fit polynomial due to NoneType object")
            return

        # Find the x and y coordinates of all the nonzero 
        # (i.e. white) pixels in the frame.         
        nonzero = self.warped_frame.nonzero()  
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Store left and right lane pixel indices
        left_lane_inds = ((nonzerox > (left_fit[0]*(
            nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0]*(
            nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(
            nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0]*(
            nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))           

        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds

        # Get the left and right lane line pixel locations  
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]  

        # Fit a second order polynomial curve to each lane line
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        self.left_fit = left_fit
        self.right_fit = right_fit

        # Create the x and y values to plot on the image
        ploty = np.linspace(
        0, self.warped_frame.shape[0]-1, self.warped_frame.shape[0]) 
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        self.ploty = ploty
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx

        if plot==True:

            # Generate images to draw on
            out_img = np.dstack((self.warped_frame, self.warped_frame, (
                                self.warped_frame)))*255
            window_img = np.zeros_like(out_img)

            # Add color to the left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                                                                            0, 0, 255]
            # Create a polygon to show the search window area, and recast 
            # the x and y points into a usable format for cv2.fillPoly()
            margin = self.margin
            left_line_window1 = np.array([np.transpose(np.vstack([
                                            left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                            left_fitx+margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([
                                            right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                            right_fitx+margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            # Plot the figures 
            figure, (ax1, ax2, ax3) = plt.subplots(3,1) # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(self.warped_frame, cmap='gray')
            ax3.imshow(result)
            ax3.plot(left_fitx, ploty, color='yellow')
            ax3.plot(right_fitx, ploty, color='yellow')
            ax1.set_title("Original Frame")  
            ax2.set_title("Warped Frame")
            ax3.set_title("Warped Frame With Search Window")
            plt.show()

    def overlay_lane_lines(self, plot=False):
        """
        Overlay lane lines on the original frame
        :param: Plot the lane lines if True
        :return: Lane with overlay
        """
        # Generate an image to draw the lane lines on 
        warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))   

        # Ensure that pts is not None
        if self.left_fitx is None or self.right_fitx is None:
            print("Could not overlay lane lines due to NoneType object")
            return None

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([
                            self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([
                            self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))


        # Ensure that pts is not None
        if pts is None:
            print("Could not overlay lane lines due to NoneType object")
            return None

        # Draw lane on the warped blank image
        cv2.fillPoly(color_warp, np.int_(pts), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective 
        # matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (
                                    self.orig_frame.shape[
                                    1], self.orig_frame.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)

        if plot==True:
            # Plot the figures 
            figure, (ax1, ax2) = plt.subplots(2,1) # 2 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            ax1.set_title("Original Frame")  
            ax2.set_title("Original Frame With Lane Overlay")
            plt.show()   

        return result  
    
    def calculate_car_position(self, print_to_terminal=False):
        """
        Calculate the position of the car relative to the center of the lane

        :param print_to_terminal: Display data to console if True
        :return: Offset from the center of the lane
        """
        if self.left_fit is None or self.right_fit is None:
            # Lane lines not detected, return a default value or handle accordingly
            return 0.0

        # Assume the camera is centered in the image.
        # Get position of car in centimeters
        car_location = self.orig_frame.shape[1] / 2 + self.car_offset

        # Find the x coordinate of the lane line bottom
        height = self.orig_frame.shape[0]
        bottom_left = self.left_fit[0] * height ** 2 + self.left_fit[1] * height + self.left_fit[2]
        bottom_right = self.right_fit[0] * height ** 2 + self.right_fit[1] * height + self.right_fit[2]

        center_lane = (bottom_right - bottom_left) / 2 + bottom_left
        center_offset = (np.abs(car_location) - np.abs(center_lane)) * self.XM_PER_PIX * 100

        if print_to_terminal:
            print(str(center_offset) + 'cm')

        self.center_offset = center_offset

        return center_offset


    def histogram_peak(self):
        """
        Get the left and right peak of the histogram
    
        Return the x coordinate of the left histogram peak and the right histogram
        peak.
        """
        midpoint = int(self.histogram.shape[0]/2)
        leftx_base = np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint
    
        # (x coordinate of left peak, x coordinate of right peak)
        return leftx_base, rightx_base
    
    def is_straight(self, printToTerminal = False):

        first_point = self.left_fitx[0]
        second_point = self.left_fitx[-1]
        if printToTerminal:
            print(f'Abstand Linien {first_point - second_point}')

        #for left curve
        if (first_point - second_point) > 0 and (first_point - second_point) < 100:

            return True
        
        else:

            False

        #if self.left_fit

def main_lanes(frame):
    """
    Perform main lane detection processes including white detection, blending out cars, region of interest (ROI) extraction,
    perspective transformation, histogram calculation, lane line detection using sliding windows, filling in the lane lines,
    overlaying lines on the original frame, and calculating the car's position offset.
    :param frame: The input frame for lane detection
    :return: Tuple containing the car's center offset and time data collected during the process
    """
    #lane detection class
    lane_detection = LaneDetection(frame)

    #timer for process time
    start_time = time.time()

    #show only white pixels in image
    white_image = lane_detection.white(frame)
    white_time = (time.time() - start_time) * 1000

    #blend out car
    #no_car = lane_detection.blend_out_car(white_image)
    no_car_time = (time.time() - start_time) * 1000

    #show region of interest by displaying them
    lane_detection.plot_region_of_interest(white_image, False)
    roi_time = (time.time() - start_time) * 1000

    #transform perspective
    cropped_image = lane_detection.perspective_transform(white_image, False)
    transform_time = (time.time() - start_time) * 1000

    cv2.imshow("warped", cropped_image)

    #histogram
    histogram = lane_detection.calculate_histogram(cropped_image, plot=False)  
    histogram_time = (time.time() - start_time) * 1000

    #show histogram
    histo = False
    if histo == True:
        # Turn on interactive mode
        plt.ion()

        # Clear any previous plot
        plt.clf()

        # Plot the histogram
        plt.plot(histogram)
        plt.title("Histogram Peaks")
        plt.show()

    # Find lane line pixels using the sliding window method 
    left_fit, right_fit = lane_detection.get_lane_line_indices_sliding_windows(plot=False)
    # Fill in the lane line
    lane_detection.get_lane_line_previous_window(left_fit, right_fit, plot=False)

    sliding_lane_time = (time.time() - start_time) * 1000
     
    # Overlay lines on the original frame
    frame_with_lane_lines = lane_detection.overlay_lane_lines(plot=False)

    overlay_time = (time.time() - start_time) * 1000

    # Check if frame_with_lane_lines is not None and has valid dimensions
    if frame_with_lane_lines is not None and frame_with_lane_lines.shape[0] > 0 and frame_with_lane_lines.shape[1] > 0:
        cv2.imshow("Frame with lane lines", frame_with_lane_lines)
    else:
        print("Error: Invalid dimensions for frame_with_lane_lines")

    # Calculate center offset                                                                 
    center_offset = lane_detection.calculate_car_position(print_to_terminal=False)

    car_position_time = (time.time() - start_time) * 1000

    straight = lane_detection.is_straight()

    if straight:
        print("Gerade")
    else:
        print("Kurve")

    

    data = [time.strftime("%H:%M:%S"),
        white_time,
        no_car_time,
        roi_time,
        transform_time,
        histogram_time,
        sliding_lane_time,
        overlay_time,
        car_position_time]

    return center_offset, data, straight