import cv2
import numpy as np
from driving import Functions_Driving
driving_instance = Functions_Driving()


#line tracking system
def tracking(image):
    while True:
        frame = cv2.resize(image, (720,480))    #resizing the camera window

        # Choosing points for perspective transformation
        tleft_tracking = (235,200)
        bleft_tracking = (65,270)
        tright_tracking = (495,200)
        bright_tracking = (655,270)
        # Creating the points
        cv2.circle(frame, tleft_tracking, 5, (0,0,255), -1)
        cv2.circle(frame, bleft_tracking, 5, (0,0,255), -1)
        cv2.circle(frame, tright_tracking, 5, (0,0,255), -1)
        cv2.circle(frame, bright_tracking, 5, (0,0,255), -1)

        # Aplying perspective transformation
        pts1 = np.float32([tleft_tracking, bleft_tracking, tright_tracking, bright_tracking])   #points original video
        pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])     #points to be (original video points) in the transformed image
        
        # Matrix to warp the image for birdseye window
        matrix = cv2.getPerspectiveTransform(pts1, pts2)    #first points "warped" to seconds points
        transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))   #matrix converted to "normal" image

        #mask on birds eye view
        hsv_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)  #make frame colors gray, so pixels are 0 or 1
        lower = np.array([0, 0, 200])
        upper = np.array([255, 50, 255])    #color to detect
        mask = cv2.inRange(hsv_frame, lower, upper)

        #mask on original frame
        hsv_frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)     #make frame colors gray, so pixels are 0 or 1
        lower2 = np.array([0, 0, 200])
        upper2 = np.array([255, 255, 255])    #color to detect
        mask2 = cv2.inRange(hsv_frame2, lower2, upper2)
        result = cv2.bitwise_and(frame, frame, mask = mask2)



        #applying boxes with a histogram
        histogram = np.sum(mask[mask.shape[0]//2:, :], axis = 0) #bottom half of image
        midpoint = int(histogram.shape[0]/2) #split image in the mid -> so left and right line are seperated
        left_base = np.argmax(histogram[:midpoint]) #store peaks of the histogram (of the left side)
        right_base = np.argmax(histogram[midpoint:]) +midpoint #store peaks of the histogram (of the right side)
        #the argmax function only takes the peak of the histogram (highest intensity). So it replaces also a noise reduction because the noises got a lower peak at the histogram.


        #sliding windows
        y = 472 #window
        lx = [] #points of left lane
        rx = [] #points of left lane
        #12 sliding windows bcs 480/40 = 12 (480 = height window, 40 = heigt of box (of sliding window))

        msk = mask.copy()


        while y>0:

            #Left threshold
            img = mask[y-40:y, left_base - 50:left_base+50] #img is only the first of the 12 boxes
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #find countours (the 1s in the img)
            for contour in contours:
                M = cv2.moments(contour) #get moments of the contour
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"]) #x coordinate of the pixels that are not 0
                    cy = int(M["m01"]/M["m00"]) #y coordinate of the pixels that are not 0
                    lx.append(left_base-50 + cx) #we only need x because later it will be 12 retangles with specified height (y)
                    left_base = left_base-50 + cx


            ## Right threshold
            img = mask[y-40:y, right_base-50:right_base+50]
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    rx.append(right_base-50 + cx)
                    right_base = right_base-50 + cx
            
            cv2.rectangle(msk, (left_base-50,y), (left_base+50,y-40), (255,255,255), 2)
            cv2.rectangle(msk, (right_base-50,y), (right_base+50,y-40), (255,255,255), 2)

            y-=40 #height of the boxes

            offset_left = midpoint-left_base #offset left
            offset_right = right_base-midpoint #offset right



        BLACK = (255,0,0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1.1
        font_color = BLACK
        font_thickness = 2
        text = str(offset_left)
        text2 = str(offset_right)
        x,y = 100,400
        img_text = cv2.putText(msk, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
        img_text2 = cv2.putText(msk, text2, (x+340,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

        if offset_left-offset_right < -60:
            img_text3 = cv2.putText(msk, "steer right", (x+340,y+20), font, font_size, font_color, font_thickness, cv2.LINE_AA)
            driving_instance.left_steering()
        elif offset_left-offset_right > 60:
            img_text3 = cv2.putText(msk, "steer left", (x,y+20), font, font_size, font_color, font_thickness, cv2.LINE_AA)
            driving_instance.right_steering()
        elif offset_left-offset_right < 60 and offset_left-offset_right > -60:
            img_text3 = cv2.putText(msk, "steer straight", (x+150,y+20), font, font_size, font_color, font_thickness, cv2.LINE_AA)
            driving_instance.neutral_steering()
    #cv2.imshow("Frame", frame)
    #cv2.imshow("result on birds eye", mask)
    #cv2.imshow("result on original", result)
    #cv2.imshow("result sliding windows", msk)

    #if cv2.waitKey(10) == 27:
    #    break

        return frame, mask, result, msk


def main_detect(frame):
    
    normal, mask, result, msk = tracking(frame)

    return normal, mask, result, msk