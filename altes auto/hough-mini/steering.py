# steering.py
import numpy as np
import math

class SteeringController:
    def __init__(self, driving_instance):
        self.driving_instance = driving_instance

    def distance_line_to_point(self, px2, py2, line_left, line_right):
        target_y = py2

        #calculate the parameters of the left line
        parameters_left = np.polyfit((line_left[0], line_left[1]), (line_left[2], line_left[3]), 1)
        slope_left = parameters_left[0]
        intercept_left = parameters_left[1]

        #calculate the x-coordinate on the left line at the target y
        x_target_left = (target_y - intercept_left) / slope_left
        y_target_left = target_y

        #calculate the perpendicular distance between the left line and the target point
        distance_left = abs(slope_left * px2 - py2 + intercept_left) / math.sqrt(slope_left**2 + 1)

        #calculate the coordinates of the closest point on the left line
        x_line_left = px2 - (distance_left * slope_left) / (slope_left**2 + 1)
        y_line_left = slope_left * x_line_left + intercept_left

        #calculate the parameters of the right line
        parameters_right = np.polyfit((line_right[0], line_right[1]), (line_right[2], line_right[3]), 1)
        slope_right = parameters_right[0]
        intercept_right = parameters_right[1]

        #calculate the x-coordinate on the right line at the target y
        x_target_right = (target_y - intercept_right) / slope_right
        y_target_right = target_y

        #calculate the perpendicular distance between the right line and the target point
        distance_right = abs(slope_right * px2 - py2 + intercept_right) / math.sqrt(slope_right**2 + 1)

        #calculate the coordinates of the closest point on the right line
        x_line_right = px2 - (distance_right * slope_right) / (slope_right**2 + 1)
        y_line_right = slope_right * x_line_right + intercept_right

        print(f"Distance left: {distance_left + 40}")
        print(f"Target point on the left line: ({x_target_left}, {y_target_left})")
        print(f"Closest point on the left line: ({x_line_left}, {y_line_left})")

        print(f"Distance right: {distance_right}")
        print(f"Target point on the right line: ({x_target_right}, {y_target_right})")
        print(f"Closest point on the right line: ({x_line_right}, {y_line_right})")

        #call the steering function to determine the direction to steer
        steering_direction = self.steer_based_on_distance_difference(
            distance_left, distance_right,
        )

        position = "left"

        print(f"Steering direction: {steering_direction}")

    def steer_based_on_distance_difference(self, distance_left, distance_right, threshold=5.0):
        #check the difference between left and right distances
        distance_left = distance_left + 40
        distance_difference = abs(distance_left - distance_right)
        max_difference = 30

        if distance_difference < threshold:
            #if the difference is small, move straight
            self.driving_instance.neutral_steering()
            steering_direction = "straight"
        elif distance_left < distance_right:
            #if the left distance is smaller, steer right
            variance = distance_difference / max_difference
            if variance <= 1:
                self.driving_instance.right_steering(variance)
            else:
                self.driving_instance.right_steering(1)

            steering_direction = "right"
        else:
            #if the right distance is smaller, steer left
            variance = distance_difference / max_difference
            if variance <= 1:
                self.driving_instance.left_steering(variance)
            else:
                self.driving_instance.left_steering(1)
            steering_direction = "left"

        return steering_direction
