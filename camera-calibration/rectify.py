import numpy as np
import cv2

data = np.load('calib_data/MultiMatrix.npz')
camMatrix = data["camMatrix"]
distCof = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]

image = cv2.imread('images/img15.jpg')

#undistore
correct_image = cv2.undistort(image, camMatrix, distCof, None, camMatrix)
cv2.imshow('correct image: ', correct_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
