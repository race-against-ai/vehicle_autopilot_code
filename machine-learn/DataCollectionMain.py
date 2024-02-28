
import DataCollectionModule as dcM
import cv2
from time import sleep
import WebcamModule as wM

from piracer.vehicles import PiRacerPro
from piracer.gamepads import ShanWanGamepad

shanwan_gamepad = ShanWanGamepad()
piracer = PiRacerPro()

record = 0

while True:

    gamepad_input = shanwan_gamepad.read_data()

    throttle = gamepad_input.analog_stick_right.y * 0.2
    steering = gamepad_input.analog_stick_left.x

    start_button =  gamepad_input.analog_stick_left.y
    print(f'Start button pressed: {start_button}')

    print(f'throttle={throttle}, steering={steering}')

    if start_button == 1:
        if record == 0: print('Recording Started ...')
        record +=1
        sleep(0.300)
    if record == 1:
        img = wM.getImg(False,size=[240,120])
        dcM.saveData(img,steering)
    elif record == 2:
        dcM.saveLog()
        record = 0

    piracer.set_throttle_percent(throttle)
    piracer.set_steering_percent(steering)
    cv2.waitKey(1)

