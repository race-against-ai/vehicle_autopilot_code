from piracer.vehicles import PiRacerStandard

battery_percent_backup = 100
status = 2 #status = 1 charging, status = 2 no charging


class Functions_Driving:
    def __init__(self):
        self.piracer = PiRacerStandard()

    #steering left
    def left_steering(self, percent):
        self.piracer.set_steering_percent(-percent)
    
    #steering right
    def right_steering(self, percent):
        self.piracer.set_steering_percent(percent)
    
    #steering Forward
    def frward_drive(self, percent):
        self.piracer.set_throttle_percent(percent)
    
    #brake
    def braking(self):
        self.piracer.set_throttle_percent(-1.0)
        self.piracer.set_throttle_percent(0.0)

    #steering Backward
    def backward(self):
        self.piracer.set_throttle_percent(-0.3)

    #stop
    def stop(self):
        self.piracer.set_throttle_percent(0.0)

    #steering neutral
    def neutral_steering(self):
        self.piracer.set_steering_percent(0.0)
