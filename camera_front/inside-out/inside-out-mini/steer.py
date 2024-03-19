class PIDController:
    def __init__(self, kp, ki, kd, target):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.target = target  # Desired position (center of the lane)
        self.prev_error = 0  # Previous error (for derivative)
        self.integral = 0  # Integral of the error

    def update(self, current_pos):
        error = current_pos - self.target  # Calculate current error
        self.integral += error  # Update integral of the error
        derivative = error - self.prev_error  # Calculate derivative
        self.prev_error = error  # Update previous error

        # Calculate control signal (steering angle)
        control_signal = self.kp * error + self.ki * self.integral + self.kd * derivative

        return control_signal


# Example usage
if __name__ == "__main__":
    # Define parameters
    kp = 0.1  # Proportional gain
    ki = 0.01  # Integral gain
    kd = 0.01  # Derivative gain
    target = 0.1  # Slightly off the center of the lane

    # Initialize PID controller
    pid_controller = PIDController(kp, ki, kd, target)

    # Simulate continuous update of car position
    for i in range(100):
        # Simulate receiving new position value for the car
        current_position = 2 - i * 0.1  # Example: decreasing position over time

        # Update control signal
        control_signal = pid_controller.update(current_position)

        # Simulate car movement (adjust based on control signal)
        # For example, decrease the current_position by the control signal to simulate the car moving towards the center of the lane
        current_position -= control_signal

        # Print current position and control signal (for demonstration)
        print("Current Position:", current_position, "Control Signal:", control_signal)


import numpy as np
from driving import Functions_Driving
driving_instance = Functions_Driving()


def steer(motor, curve, center_offset):

    #calculation for steering angle
    angle = calculate_steering_angle(center_offset, curve)

    if angle == 0:
        driving_instance.neutral_steering()
    elif angle < 0:
        driving_instance.left_steering(angle)
    else:
        driving_instance.left_steering(angle)

    if motor:
        if curve:
            driving_instance.frward_drive(0.2)
        else:
            driving_instance.frward_drive(0.2)
    else:
        driving_instance.frward_drive(0)
            
            
def calculate_steering_angle(center_offset, curve):
    """
    Calculate steering angle from offset to middle
    :param offset: Offset from the middle (-250 to 250)
    :return: Steering angle in range [-1, 1]
    """
    if abs(center_offset) < 0:
        return 0.0

    if curve:
        normalized_offset = center_offset / 70
    else:
        normalized_offset = center_offset / 160

    normalized_offset *= -1

    angle = np.clip(normalized_offset, -0.8, 0.8)

    print(f'Steering angle: {angle}')

    return angle