import math

class Vector:

    def __init__(self):

        self._debug = None
        # start points
        self._point1 = None
        self._point2 = None

        #point to check
        self._point_to_check = None

        # offset and slope of the first line
        self._line_intercept = None
        self._line_slope = None

        #orthogonal
        self._orthogonal_slope = None
        self._orthogonal_intercept = None

        #insersection point
        self._x_intersection = None
        self._y_intersection = None
        

    def calculate_distance(self, point1, point2, point_to_check, debug = False):
        self._point1 = point1
        self._point2 = point2
        self._point_to_check = point_to_check

        self._straight_line_equation_with_two_points(debug)

        self._orthogonal_line_to_line(debug)

        self._intersection_point_between_lines(debug)

        distance = self._distance_between_two_points()

        return distance

    def _straight_line_equation_with_two_points(self, debug = False):
        x1, y1 = self._point1
        x2, y2 = self._point2

        # Calculate the slope m
        if x2 - x1 != 0:
            m = (y2 - y1) / (x2 - x1)
        else:
            # Handle the case of a vertical line
            m = float('inf')

        # Calculate the y-intercept b
        b = y1 - m * x1

        # Return the equation
        if debug:
            print(f'y = {m}x + {b}')

        self._line_intercept = b
        self._line_slope = m
    
    def _orthogonal_line_to_line(self, debug = False):
        # Calculate the slope of the perpendicular line
        if self._line_slope != 0:  # Special case when the slope of the given line is not zero
            orthogonal_slope = -1 / self._line_slope
        else:
            orthogonal_slope = float('inf')  # The slope of a perpendicular line is infinity

        # Calculate the y-intercept b of the perpendicular line
        x_point, y_point = self._point_to_check
        orthogonal_intercept = y_point - orthogonal_slope * x_point

        # Return the equation of the perpendicular line
        if debug:
            print(f'y = {orthogonal_slope}x + {orthogonal_intercept}')

        self._orthogonal_slope = orthogonal_slope
        self._orthogonal_intercept = orthogonal_intercept


    def _intersection_point_between_lines(self, debug = False):
        # Calculate the x-coordinate of the intersection point
        if self._line_slope != 0:  # Special case when the slope of the given line is not zero
            x_intersection = (self._orthogonal_intercept - self._line_intercept) / (self._line_slope - self._orthogonal_slope)
        else:
            x_intersection = (self._orthogonal_intercept - self._line_intercept)  # x-coordinate of the intersection point is the y-intercept of the given line

        # Calculate the y-coordinate of the intersection point
        y_intersection = self._line_slope * x_intersection + self._line_intercept

        if debug:
            print(f'x: {x_intersection}, y: {y_intersection}')

        self._x_intersection = x_intersection
        self._y_intersection = y_intersection

    def _distance_between_two_points(self):
        x1, y1 = self._point_to_check
        x2, y2 = self._x_intersection, self._y_intersection

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance


