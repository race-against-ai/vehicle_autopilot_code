from piracer.vehicles import PiRacerPro

class Functions_Driving:
    def __init__(self):
        self.piracer = PiRacerPro()

    # Steering left
    def left_steering(self):
        self.piracer.set_steering_percent(0.5)
    
    # Steering right
    def right_steering(self):
        self.piracer.set_steering_percent(-0.5)
    
    # Steering Forward
    def frward_drive(self):
        self.piracer.set_throttle_percent(0.1)
    
    # Brake
    def braking(self):
        self.piracer.set_throttle_percent(-1.0)
        self.piracer.set_throttle_percent(0.0)

    # Steering Backward
    def backward(self):
        self.piracer.set_throttle_percent(-0.3)

    # Stop
    def stop(self):
        self.piracer.set_throttle_percent(0.0)

    # Steering neutral
    def neutral_steering(self):
        self.piracer.set_steering_percent(0.0)
