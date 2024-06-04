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
        

    def _straight_line_equation_with_two_points(self, debug = False):
        x1, y1 = self._point1
        x2, y2 = self._point2

        # Calculate the slope m
        if x2 - x1 != 0:
            m = (y2 - y1) / (x2 - x1)
        else:
            # Handle the case of a vertical line
            m = 1e9

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


    def _intersection_point_between_lines(self, debug=False):
        if self._line_slope == float('inf'):  # Special case for a vertical line
            x_intersection = self._point_to_check[0]  # x-coordinate of the intersection is the x-coordinate of the vertical line
            y_intersection = self._line_intercept  # y-coordinate of the intersection is the y-intercept of the vertical line
        else:
            # Calculate the x-coordinate of the intersection point
            x_intersection = (self._orthogonal_intercept - self._line_intercept) / (self._line_slope - self._orthogonal_slope)

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
    
    def _left_or_right(self):

        x1 = self._point_to_check[0]
        x2 =  self._x_intersection

        print(x1, x2)

        if abs(x1) > 1 and abs(x2) > 1:

            if x1 < x2:
                return "left"
            else:
                return "right"


    def calculate_distance(self, point1, point2, point_to_check, curve_check = False ,debug = False):
        self._point1 = point1
        self._point2 = point2
        self._point_to_check = point_to_check

        self._straight_line_equation_with_two_points(debug)

        self._orthogonal_line_to_line(debug)

        self._intersection_point_between_lines(debug)

        distance = self._distance_between_two_points()

        if curve_check:
            curve_direction = self._left_or_right()
            return curve_direction

        return distance