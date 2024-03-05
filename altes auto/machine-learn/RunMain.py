import cv2
import numpy as np
from tensorflow.keras.models import load_model

import WebcamModule as wM

from piracer.vehicles import PiRacerPro
piracer = PiRacerPro()

#import WebcamModule as wM
#import MotorModule as mM

#######################################
#steeringSen = 0.70  # Steering Sensitivity
maxThrottle = 0.2 # Forward Speed %
#motor = mM.Motor(2, 3, 4, 17, 22, 27) # Pin Numbers
model = load_model('/home/itlab/cam/model.h5')
######################################

def preProcess(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

while True:

    img = wM.getImg(False, size=[240, 120])
    img = np.asarray(img)
    img = preProcess(img)
    img = np.array([img])
    steering = float(model.predict(img))
    print(steering)

    piracer.set_steering_percent(steering)
    piracer.set_throttle_percent(maxThrottle)

    cv2.waitKey(1)
