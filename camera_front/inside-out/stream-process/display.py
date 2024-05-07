import cv2
import queue
import numpy as np

class FourFrameWindow:
    def __init__(self):
        self.frames = {} 

    def display(self, frames, data_car, count, *frame_names):
        self.frames = frames
        selected_frames = [self.frames[frame_name] for frame_name in frame_names]

        if count == 2:
            self._display_two_frames(selected_frames)
        elif count == 3:
            self._display_three_frames(selected_frames, data_car)
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

    def _display_three_frames(self, frames, data_car):
        resized_frames = [cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) for frame in frames]

        # Convert all frames to RGB if they are in grayscale
        for i, frame in enumerate(resized_frames):
            if len(frame.shape) == 2:
                resized_frames[i] = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Concatenate the first two frames horizontally
        im_h_resize_top = self.hconcat_resize_min([resized_frames[0], resized_frames[1]])

        info_frame = information_window(resized_frames[2], data_car)
        im_h_resize_bottom = info_frame

        # Concatenate both rows vertically
        im_v_resize = cv2.vconcat([im_h_resize_top, im_h_resize_bottom])

        # Display the concatenated frames in one window
        cv2.imshow("Concatenated Frames", im_v_resize)


    @staticmethod
    def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
        h_min = min(im.shape[0] for im in im_list)
        im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                        for im in im_list]
        return cv2.hconcat(im_list_resize)


def information_window(frame_to_add, data_car):

    # Create a blank white image with dimensions 1024x384
    window = 255 * np.ones((384, 1024, 3), dtype=np.uint8)

    frame = frame_to_add

    distance_left, distance_right, info_left, info_right, speed, curve, distance_speed, distance_curve = data_car

    distance_speed = round(distance_speed, 2)
    distance_curve = round(distance_curve, 2)

    # Add text on the left and right sides
    text_left_line1 = f"{info_left}"
    text_left_line2 = f"left space: {distance_left}"
    text_left_line3 = f"brake: {speed}"
    text_left_line4 = f"in curve: {curve}"
    text_right_line1 = f"{info_right}"
    text_right_line2 = f"right space: {distance_right}"
    text_right_line3 = f"offset brake: {distance_speed}"
    text_right_line4 = f"offset curve: {distance_curve}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    # Get text sizes
    text_size_left_line1 = cv2.getTextSize(text_left_line1, font, font_scale, font_thickness)[0]
    text_size_left_line2 = cv2.getTextSize(text_left_line2, font, font_scale, font_thickness)[0]
    text_size_left_line3 = cv2.getTextSize(text_left_line3, font, font_scale, font_thickness)[0]
    text_size_left_line4 = cv2.getTextSize(text_left_line4, font, font_scale, font_thickness)[0]
    text_size_right_line1 = cv2.getTextSize(text_right_line1, font, font_scale, font_thickness)[0]
    text_size_right_line2 = cv2.getTextSize(text_right_line2, font, font_scale, font_thickness)[0]
    text_size_right_line3 = cv2.getTextSize(text_right_line3, font, font_scale, font_thickness)[0]
    text_size_right_line4 = cv2.getTextSize(text_right_line4, font, font_scale, font_thickness)[0]

    # Calculate maximum width of right-side text
    max_right_text_width = 230

    # Calculate positions for the text with increased spacing
    space_between_lines = 30
    text_pos_left_line1 = (10, window.shape[0] // 4 + text_size_left_line1[1] // 2)
    text_pos_left_line2 = (10, window.shape[0] // 4 + text_size_left_line1[1] + space_between_lines + text_size_left_line2[1] // 2)
    text_pos_left_line3 = (10, window.shape[0] // 4 + text_size_left_line1[1] + space_between_lines * 2 + text_size_left_line2[1] + text_size_left_line3[1] // 2)
    text_pos_left_line4 = (10, window.shape[0] // 4 + text_size_left_line1[1] + space_between_lines * 3 + text_size_left_line2[1] + text_size_left_line3[1] + text_size_left_line4[1] // 2)
    text_pos_right_line1 = (window.shape[1] - max_right_text_width - 10, window.shape[0] // 4 + text_size_right_line1[1] // 2)
    text_pos_right_line2 = (window.shape[1] - max_right_text_width - 10, window.shape[0] // 4 + text_size_right_line1[1] + space_between_lines + text_size_right_line2[1] // 2)
    text_pos_right_line3 = (window.shape[1] - max_right_text_width - 10, window.shape[0] // 4 + text_size_right_line1[1] + space_between_lines * 2 + text_size_right_line2[1] + text_size_right_line3[1] // 2)
    text_pos_right_line4 = (window.shape[1] - max_right_text_width - 10, window.shape[0] // 4 + text_size_right_line1[1] + space_between_lines * 3 + text_size_right_line2[1] + text_size_right_line3[1] + text_size_right_line4[1] // 2)

    # Put text on the window
    cv2.putText(window, text_left_line1, text_pos_left_line1, font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(window, text_left_line2, text_pos_left_line2, font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(window, text_left_line3, text_pos_left_line3, font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(window, text_left_line4, text_pos_left_line4, font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(window, text_right_line1, text_pos_right_line1, font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(window, text_right_line2, text_pos_right_line2, font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(window, text_right_line3, text_pos_right_line3, font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(window, text_right_line4, text_pos_right_line4, font, font_scale, (0, 0, 0), font_thickness)

    # Display the frame in the middle
    window[0:frame.shape[0], (window.shape[1] - frame.shape[1]) // 2:(window.shape[1] + frame.shape[1]) // 2] = frame

    return window


def displayImages(four_frame_window, data_car, frames):
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
    four_frame_window.display(frames, data_car, 3, 'original', 'steering' ,'lanes')#, 'steering', 'time')