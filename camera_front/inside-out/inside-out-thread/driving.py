from piracer.vehicles import PiRacerStandard

from adafruit_ssd1306 import SSD1306_I2C
import board
import busio

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

    def battery(self):

        battery_voltage = self.piracer.get_battery_voltage()
        battery_current = self.piracer.get_battery_current()
        power_consumption = self.piracer.get_battery_voltage()

    def battery_percent(self):
        global battery_percent_backup
        global status

        percent = []
        trigger_charging = False
        trigger_nocharging = False

        for i in range(0, 3, 1): 

            #get the current battery voltage
            battery_voltage = self.piracer.get_battery_voltage()
            battery_current = self.piracer.get_battery_current()


            if battery_voltage > 8 or battery_current > 100 :
                if status == 2:
                    trigger_charging = True
                    status = 1
                charging = True
                #assume a maximum battery voltage and a minimum voltage for estimation
                max_voltage = 8.31  #replace with the maximum voltage of your battery
                min_voltage = 6.835990  #replace with the minimum voltage of your battery

                #calculate the battery percentage (rounded up to the nearest ten)
                battery_percent = int(((battery_voltage - min_voltage) / (max_voltage - min_voltage)) * 100.0 + 0.5)
                if battery_percent > 100:
                    battery_percent = 100
            else:
                if status == 1:
                    trigger_nocharging = True
                    status = 2
                charging = False
                #assume a maximum battery voltage and a minimum voltage for estimation
                max_voltage = 7.8  #replace with the maximum voltage of your battery
                min_voltage = 6.835990  #replace with the minimum voltage of your battery

                #calculate the battery percentage (rounded up to the nearest ten)
                battery_percent = int(((battery_voltage - min_voltage) / (max_voltage - min_voltage)) * 100.0 + 0.5)
                if battery_percent > 100:
                    battery_percent = 100

            percent.append(battery_percent)

        #calculate the mean
        mean_percent = int(sum(percent) / len(percent))


        ##if charging show charging... else show battery percentage
        if charging:
            if mean_percent < battery_percent_backup-2 or mean_percent > battery_percent_backup+1 or trigger_charging:
                trigger_charging = False
                text = f"Charging...\n{mean_percent}%"
                self.display(text)
                battery_percent_backup = mean_percent

            charging = False
        elif mean_percent < battery_percent_backup-1 or mean_percent > battery_percent_backup+3 or trigger_nocharging:
            trigger_nocharging = False
            text = f"Battery: {mean_percent}%"
            self.display(text)
            battery_percent_backup = mean_percent


    def display(self, text_display):

        #initialize the I2C bus
        SCL = board.SCL
        SDA = board.SDA
        i2c_bus = busio.I2C(SCL, SDA)

        #initialize the SSD1306 display
        display = SSD1306_I2C(128, 32, i2c_bus, addr=0x3C)

        #clear the display
        #display.fill(0)
        #display.show()

        #write text to the display
        display.text(text_display, 0, 0, 1)  # The last argument (1) is the color (1 for white, 0 for black)

        #show the updated display
        display.show()