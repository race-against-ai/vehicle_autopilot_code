from prettytable import PrettyTable
import time

class Timer:

    def __init__(self):

        self.total_time_sum = 0
        self.total_fps_sum = 0
        self.iterations = 1
        self.min_fps = 100

        #time data
        self.data = None

        self.start_time = None

    def print_table(self, data, time):

        self.data = data
        self.start_time = time
                    
        total_time, fps, average_time, average_fps, min_fps = self.calculate_process_time()
        self.data = self.update_data(self.data, total_time, fps, average_time, average_fps, min_fps)

        table = PrettyTable()
        table.field_names = ["Rows selection", "Transform", "Sliding Lane", "Car Position" , "Curve Detection", "Total Main Loop", "FPS", "Average Time", "Average FPS", "min FPS"]
        table.add_row(self.data)

        print(table)

        self.iterations += 1


    def calculate_process_time(self):
        total_time = (time.time() - self.start_time) * 1000
        fps = 1000 / total_time
        self.total_fps_sum += fps
        if fps < self.min_fps and self.iterations > 50:
            self.min_fps = fps
        self.total_time_sum += total_time
        average_time = self.total_time_sum / self.iterations
        average_fps = self.total_fps_sum / self.iterations
        return total_time, fps, average_time, average_fps, self.min_fps

    @staticmethod
    def update_data(data, total_time, fps, average_time, average_fps, min_fps):
        data.append(total_time)
        data.append(fps)
        data.append(average_time)
        data.append(average_fps)
        data.append(min_fps)
        return data