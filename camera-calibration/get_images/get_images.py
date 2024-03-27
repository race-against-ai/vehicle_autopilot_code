import cv2

image_count = 1

def capture_image():
    global image_count
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

    #image
    ret, frame = cap.read()
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)

    #save
    image_name = f'img{image_count}.jpg'
    cv2.imwrite(image_name, rotated_frame)
    print(f"captured: {image_name}")

    image_count += 1

    cap.release()

def main():
    while True:
        key = input("'s' to save or 'q' to exit: ")
        if key == 's':
            capture_image()
        elif key == 'q':
            print("finished")
            break

if __name__ == "__main__":
    main()
