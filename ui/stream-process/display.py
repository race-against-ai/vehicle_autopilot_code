import cv2
import queue

class FourFrameWindow:
    def __init__(self):
        self.frames = {}


    def display(self, count, *frame_names):
        self.frames = getQueue()
        selected_frames = [self.frames[frame_name] for frame_name in frame_names]

        if count == 2:
            self._display_two_frames(selected_frames)
        elif count == 3:
            self._display_three_frames(selected_frames)
        elif count == 4:
            self._display_four_frames(selected_frames)
        else:
            print("Invalid count. Count must be 2 or 4.")

    def _display_four_frames(self, frames):
            
        # Resize frames by 1/4
        frame1_resized = cv2.resize(frames[0], (0, 0), fx=0.25, fy=0.25)
        frame2_resized = cv2.resize(frames[1], (0, 0), fx=0.25, fy=0.25)
        frame3_resized = cv2.resize(frames[2], (0, 0), fx=0.25, fy=0.25)
        frame4_resized = cv2.resize(frames[3], (0, 0), fx=0.25, fy=0.25)

        # Concatenate resized frames horizontally
        top_row = cv2.hconcat([frame1_resized, frame2_resized])
        bottom_row = cv2.hconcat([frame3_resized, frame4_resized])

        # Concatenate top and bottom rows vertically
        output_frame = cv2.vconcat([top_row, bottom_row])

        # Display the output frame
        cv2.imshow('Resized Frames', output_frame)

    def _display_two_frames(self, frames):
        
        resized_frames = [cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) for frame in frames]

        #combined_frame = cv2.hconcat([resized_frames[0], resized_frames[1]])

        cv2.imshow('Two Frame', resized_frames[0])
        cv2.imshow('Two Frame Window', resized_frames[1])

        resized_frames[1] = cv2.cvtColor(resized_frames[1], cv2.COLOR_GRAY2RGB)  # Assuming BGR input

        print("Resized Frame 1 shape:", resized_frames[0].shape, "Resized Frame 2 shape:", resized_frames[1].shape)

        im_h_resize = self.hconcat_resize_min([resized_frames[0], resized_frames[1]])

        cv2.imshow("imh", im_h_resize)

    def _display_three_frames(self, frames):
        resized_frames = [cv2.resize(frame, (0, 0), fx=0.7, fy=0.7) for frame in frames]

        # Convert all frames to RGB if they are in grayscale
        for i, frame in enumerate(resized_frames):
            if len(frame.shape) == 2:
                resized_frames[i] = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Display individual resized frames
        cv2.imshow('Frame 1', resized_frames[0])
        cv2.imshow('Frame 2', resized_frames[1])
        cv2.imshow('Frame 3', resized_frames[2])

        # Concatenate frames horizontally
        im_h_resize = self.hconcat_resize_min([resized_frames[0], resized_frames[1], resized_frames[2]])

        # Display the concatenated frame
        cv2.imshow("Concatenated Frame", im_h_resize)


    @staticmethod
    def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
        h_min = min(im.shape[0] for im in im_list)
        im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                        for im in im_list]
        return cv2.hconcat(im_list_resize)

image_queue = queue.Queue(maxsize=1)

def isFull():
    if image_queue.full() == True:
        return True
    else:
        return False

def putQueue(frames):
    if not isFull():
        image_queue.put(frames)

def getQueue():
    print(image_queue)
    frames = image_queue.get(block=True)

    table_image, steering_wheel, orig_frame, detected_lanes = frames[0], frames[1], frames[2], frames[3]

    frames = {
        "original": orig_frame,
        "lanes": detected_lanes,
        "steering": steering_wheel,
        "time": table_image
    }

    return frames



def displayImages(four_frame_window, data_car):
    # Example usage:

    # Assuming you have another class called FrameProvider which provides frames
    # class FrameProvider:
    #     @staticmethod
    #     def get_frame(name):
    #         # Implement the method to provide frames based on name
    #         # For example, you can load frames from files or capture from a camera
    #         pass

    # Initialize FourFrameWindow object

    # Assuming you have frames loaded from FrameProvider
    # frame_car = FrameProvider.get_frame('car')
    # frame_simulated = FrameProvider.get_frame('simulated')

    # Add frames to FourFrameWindow
    # four_frame_window.add_frame('car', frame_car)
    # four_frame_window.add_frame('simulated', frame_simulated)

    # Display two frames labeled 'car' and 'simulated'
    four_frame_window.display(3, 'original', 'lanes', 'steering')#, 'steering', 'time')