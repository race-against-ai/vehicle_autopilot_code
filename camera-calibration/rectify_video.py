import numpy as np
import cv2

data = np.load('calib_data/MultiMatrix.npz')
camMatrix = data["camMatrix"]
distCof = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]

#image = cv2.imread('images/img8.jpg')

while True:
    cap = cv2.VideoCapture('video/output_video.avi')
    _,frame = cap.read()
    frame = cv2.resize(frame, (1024, 768))
    #undistore
    correct_image = cv2.undistort(frame, camMatrix, distCof, None, camMatrix)
    cv2.imshow('correct image: ', correct_image)
    cv2.waitKey(0)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
