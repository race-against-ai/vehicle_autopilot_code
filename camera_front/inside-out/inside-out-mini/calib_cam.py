import numpy as np
import cv2

data = np.load('calib_data/MultiMatrix.npz')
camMatrix = data["camMatrix"]
distCof = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]

def calib(frame):

    #undistore
    correct_image = cv2.undistort(frame, camMatrix, distCof, None, camMatrix)

    return correct_image
