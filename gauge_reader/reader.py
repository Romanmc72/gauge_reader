#!/usr/bin/env python3
"""
Description
-----------
This is the module that holds the class for storing the main gauge reading
program.

You can use the program like so:

```
from gauger_reader.reader import GaugeReader

image_file = "/tmp/image.jpeg"
gauge_reader = GaugeReader(image_file=image_file)
temp = gauge_reader.read_gauge()

print(temp)
```

It will return None if the image is unreadable or the needle detected is out
of bounds of the acceptable range of angles.
"""
import cv2
from numpy import pi
from numpy import cos
from numpy import sin
from numpy import arctan as atan

import trig
import constants


class GaugeReader:
    """
    Description
    -----------
    This is the class that contains methods and variables related to the
    picture of the gauge and the objects detected therein.

    Params
    ------
    :image_file: str = None
    The image file that stores the picture of the gauge.

    :verbose: bool = False
    Whether or not to excessively print out all of the calculations.

    :min_value: int = 0
    The minimum value on the gauge.

    :max_value: int = 0
    The maximum value on the gauge.

    # NOTE For the angles measured here it IS from the persepctive of the 
    :min_angle: int = 0
    The minimum angle on the gauge associated to the minimum value.

    :max_angle: int = 0
    The maximum angle on the gauge associated to the maximum value.

    :circle_blur: int = 71
    :line_blur: int = 31
    :cache_size: int = 10
    :needle_angle: int = constants.VARIANCE

    # TODO: Create this...
    Optionally you can pass the bytes for the image into the @classmethod
    called gauge_reader_from_bytes() and get the object back.
    """
    def __init__(self,
                 image_file: str = None,
                 verbose: bool = False,
                 min_value: int = 0,
                 max_value: int = 0,
                 min_angle: int = 0,
                 max_angle: int = 0,
                 circle_blur: int = 71,
                 line_blur: int = 31,
                 cache_size: int = 10,
                 needle_angle: int = constants.VARIANCE,
    ):
        self.image_file = image_file
        self.verbose = verbose
        self.min_value = min_value
        self.max_value = max_value
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.circle_blur = circle_blur
        self.line_blur = line_blur
        self.cache_size = cache_size
        self.needle_angle = needle_angle

        # Load the image file if it exists otherwise set it to None.
        if self.image_file:
            if self.verbose:
                print(f"Reading image {args.image_name}...")
            self.image = cv2.imread(self.image_file, 0)
            self.height, self.width = self.image.shape[:2]
        else:
            self.image = None

        # Setting these for later.
        self.doctored_image = None
        self.lines = []
        self.points = []
        self.radius = None
        self.circle_center = (None, None)
        self.needle_point = (None, None)
        self.gauge_reading = None
        self.cache = deque()

    @property
    def average_cache_value(self) -> int:
        """Returns the closest int to the average value in the cache."""
        if len(self.cache) == 0:
            return 0
        else:
            return sum(self.cache) // len(self.cache)

    @property
    def number_of_lines(self):
        """How many lines did we find in the picture?"""
        return len(self.lines)

    def _doctor_image(self, blur_factor: int) -> None:
        """Blurs the image. Higher blur factor is more blurry. Odd numbers only."""
        if self.verbose:
            print("Doctoring it...")
        self.doctored_image = cv2.medianBlur(self.image, blur_factor)

    def _get_best_circle(self) -> None:
        """
        Find the best circular object present in a given the image set the
        values for its center cooridnates and its radius.
        """
        # Assuming that the circle is at largest taking up the whole screen and at
        # smallest, half the screen size.
        min_pixel_size = min(self.height, self.width)

        min_radius = min_pixel_size // 4
        max_radius = min_pixel_size // 2

        circles = cv2.HoughCircles(
            image=self.doctored_image,
            method=cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        circles = numpy.uint16(numpy.around(circles))

        if self.verbose:
            print(f"There are {len(circles)} circles")

        # Just taking the first one
        X, Y, radius = circles[0, 0]

        self.circle_center = (int(X), int(Y))
        self.radius = int(radius)

    def _mask_circle(self) -> None:
        """Mask out the circle from the image."""
        circle_img = numpy.zeros((self.height, self.width), numpy.uint8)
        cv2.circle(circle_img, self.circle_center, self.radius, 255, thickness=-1)
        self.doctored_image = cv2.bitwise_and(self.doctored_image, self.doctored_image, mask=circle_img)

    def _line_belongs(self, rho, theta)  -> bool:
        """Find out whether or not a detected line fits within the range of acceptable angles or locations."""
        # If it does not pass within a certain angle and distance of the
        # center, ignore it otherwise call it good.
        if rho > (self.circle_center[1] + DISTANCE_THRESHOLD) and (IGNORE_MIN_ANGLE_RADIANS <= theta <= IGNORE_MAX_ANGLE_RADIANS):
            skipped_lines += 1
            if self.verbose:
                print(f"not the needle: rho {rho:.3f} theta {theta:.3f}")
            return False
        # If this line already exists ignore it
        elif any([(line[1] - self.needle_angle) <= theta <= (line[1] + self.needle_angle) for line in self.lines]):
            skipped_lines += 1
            if self.verbose:
                print(f"Same line: rho {rho:.3f} theta {theta:.3f}")
            return False
        else:
            if self.verbose:
                print(f"Good line!: rho {rho:.3f} theta {theta:.3f}")
            return True

    def _get_best_lines(self):
        """Get the rho and theta values for the lines that are found in the image."""
        self.lines = []
        self.doctored_image = cv2.Canny(self.doctored_image, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(self.doctored_image, 1, DEGS_TO_RADS, 200)

        skipped_lines = 0
        if self.verbose:
            print(f"There are {len(lines)} lines.")

        for line in lines:
            rho, theta = line[0]

            if self._line_belongs(rho, theta):
                self.lines.append((int(rho), float(theta)))
            else:
                skipped_lines += 1
                continue

            if len(self.lines) >= LINES_TO_RENDER:
                break

        if self.verbose:
            print(f"Skipped {skipped_lines} line{'s' if skipped_lines != 1 else ''}")

    def _get_line_points(self) -> None:
        """
        Description
        -----------
        Gimme dat rho and theta and you can have a line!

        Well, i mean you would have 2 points on the line where it will intercept
        the x and y axes. However, some lines intercepts are not in the first
        quadrant so I will have to adjust for that. That is why you need to give
        me the screen dimensions as well.

        # TODO: Adjust for that :point_up: ^^^
        """
        self.points = []
        for rho, theta in self.lines:
            # At pi / 2 cos(theta) = 0 making rho / cos(theta) undefined. This is the
            # case of a horizontal line making the x constant and the y values a
            # function of the intercepts on the screen border at that constant x value
            if theta == pi / 2:
                x1 = 0
                y1 = rho
                x2 = self.width
                y2 = rho
            # same as the above comment, but sin(theta) = 0 on 0, pi, and 2 * pi and
            # in this case it is a vertical line not a horizontal one.
            elif theta % pi == 0:
                x1 = rho
                y1 = 0
                x2 = rho
                y2 = self.height
            else:
                x1 = 0
                y1 = rho / sin(theta)
                x2 = rho / cos(theta)
                y2 = 0
            self.points.append((int(x1), int(y1)), (int(x2), int(y2)))

    def _get_intersect_point(self):
        """
        Description
        -----------
        Given two coordinate pairs, return a point of intersection.

        Calculates things based on some middle school algebra principles of
            y = m * x + i
        to get the equation for each line and output from those equations the
        intersection. If the two lines are parallel it will return None.
        """
        line_one_points = self.points[0]
        line_two_points = self.points[1]

        m1, i1 = trig.get_line_equation(*line_one_points)
        m2, i2 = trig.get_line_equation(*line_two_points)

        # In the case either of the lines are vertical or both are parallel.
        if m1 == m2:
            self.needle_point = None, None 
        elif m1 is None:
            x_intersect = line_one_points[0]
            y_intersect = m2 * x_intersect + i2
            self.needle_point =  int(x_intersect), int(y_intersect)
        elif m2 is None:
            x_intersect = line_two_points[0]
            y_intersect = m1 * x_intersect + i1
            self.needle_point = int(x_intersect), int(y_intersect)

        # Otherwise if they are not parallel or vertical
        else:
            x_intersect = (i2 - i1)/(m1 - m2)
            y_intersect = m1 * x_intersect + i1
            self.needle_point = int(x_intersect), int(y_intersect)

    def _get_circle_intersects(self):
        """
        Description
        -----------
        This will intake the line's rho and theta as well as the circle's center
        and radius and return the tuple of 2 (x, y) cooridnate pairs corresponding
        to the intersects on the circle's perimeter with the line. Whether or not
        the line actually intersects the circle, this equation will align the line
        and the circle to the origin off the image (0, 0) calculate the intersects
        based solely on the angle and radius, then return the points offset by the
        circle's center point.
        """
        # Because this is how I wrote the math with center = (h, k).
        h = self.circle_center[0]
        k = self.circle_center[1]

        rho = self.lines[0][0]
        theta = self.lines[0][1]
        # I will actually translate it so that the circle is centered on the
        # origin and so is the line, making the equation soooo much easier
        # to calculate.

        # tan(theta) is undefined at pi / 2 so in that case we return early
        # because the line is vertical and the calculations below break.
        if theta == pi / 2:
            # In this case we just assume it cuts through the center of the circle
            # and comes out at the top and bottom exactly the radius distance from
            # the center.
            point_a = (int(h), int(k + self.radius))
            point_b = (int(h), int(k - self.radius))
            return (point_a, point_b)
        else:
            x1 = h + radius * sin(theta)
            x2 = h - radius * sin(theta)
            y1 = k - radius * cos(theta)
            y2 = k + radius * cos(theta)

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            self.needle_point = ((int(x1), int(y1)), (int(x2), int(y2)))
        self._determine_feasible_point()
    
    def _determine_feasible_point(self):
        """A circle and a line have 2 intersect points in this scenario, only one can be the needle."""
        for point in self.needle_point:


    def _determine_quadrant(self):
        """
        Determine the quadrant of the needle tip based on its location.
        Quadrants are like so:
         /--+--\\
        | 2 | 1 |
        +---+---+
        | 3 | 4 |
         \\--+--/
        But from the perspective of the screen's top left, so really rotated
        upside down from what you're seeing.
        """
        needle_x = self.needle_point[0]
        needle_y = self.needle_point[1]
        circle_x = self.circle_center[0]
        circle_y = self.circle_center[1]

        if needle_x > circle_x:
            if needle_y >= circle_y:
                self.quadrant = 2
            elif needle_y < circle_y:
                self.quadrant = 3
        elif needle_x < circle_x:
            if needle_y >= circle_y:
                self.quadrant = 1
            elif needle_y < circle_y:
                self.quadrant = 4
        elif needle_x == circle_x:
            if needle_y >= circle_y:
                self.quadrant = 1
            elif needle_y < circle_y:
                self.quadrant = 4
            elif needle_y == circle_y:
                raise ValueError("Those are the same point.")

    def _value_from_points(self):
        """Calculate the value for this particular gauge given the point and the circle's center"""
        self._determine_quadrant()

        raw_angle = -atan(
            (self.circle_center[1] - self.needle_point[1])
            /
            (self.circle_center[0] - self.needle_point[0])
        )

        if self.verbose:
            print(f"The raw angle: {raw_angle * RADS_TO_DEGS}")
            print(f"The quadrant : {self.quadrant}")

        angle_adjustment = {
            1: 0,
            2: -180,
            3: -180,
            4: 0
        }

        adjusted_angle = raw_angle * RADS_TO_DEGS + angle_adjustment[self.quadrant]
        if self.verbose:
            print(f"Adjusted angle: {adjusted_angle}")

        m, i = trig.get_line_equation(point_a=(self.min_angle, self.min_value), point_b=(self.max_angle, self.max_value))
        value_calc = m * adjusted_angle + i
        if self.verbose:
            print(f"Maybe this is the value: {value_calc:.2f}?")
        self.gauge_reading = int(value_calc)

    def _cache_value(self, value: int):
        """Add a new value to the cache of values."""
        if len(self.cache) >= self.cache_size:
            self.cache.pop()
        self.cache.appendleft(self.gauge_reading)

    def _identify_lines(self):
        """Using the doctored and masked image, find lines that work for the gauge."""
        self._get_best_lines()
        self._get_line_points()

    def read_gauge(self, output_name: str = None, cache_value: bool = True) -> int:
        """
        Description
        -----------
        Executes the main program. Processes a picture of a gauge and spits out a
        temperature in degrees fahrenheit.

        Params
        ------
        :image_name: str
        The name of the file on disk representing the image to process.

        :output_name: str = None
        Where on disk to save the final output image. If none specified then the image is not saved.

        :verbose: bool = False
        If True, will show stages of the image being processed while awaiting
        keyboard input before proceeding and will likewise print out helpful
        calculations along the way.

        Return
        ------
        int
        The temperature in degrees fahrenheit.
        """
        # Identify the circle and store its values.
        self._doctor_image(self.circle_blur)
        self._get_best_circle()

        # Mask out the circle and identify any lines.
        self._mask_circle()
        self._doctor_image(self.line_blur)
        self._identify_lines()
        
        if self.number_of_lines == 1:
            self._get_circle_intersects()
        elif self.number_of_lines == LINES_TO_RENDER:
            self._get_intersect_point()
        else:
            return None

        for p1, p2, angle, rho, theta in points:
            cv2.line(self.image, p1, p2, (255, 255, 255), 10)
            cv2.putText(
                self.image,
                f"Line Angle: {angle:3.2f}",
                p2,
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 255, 255),
                10
            )
            cv2.putText(
                self.image,
                f"Rho: {rho}",
                (p2[0], p2[1] + TEXT_HEIGHT),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 255, 255),
                10
            )
            cv2.putText(
                self.image,
                f"Theta: {theta:3.2f}",
                (p2[0], p2[1] + (TEXT_HEIGHT * 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 255, 255),
                10
            )
        if len(points) >= 2:
            points_of_interest = [p[0:2] for p in points[0:2]]
            line_intersect = get_intersect_points(*points_of_interest)
            intersections = [line_intersect]
        elif len(points) == 1:
            p1, p2, angle, rho, theta = points[0]
            intersections = get_circle_intersects(center=(x, y), radius=radius, rho=rho, theta=theta)

        offset = 100

        degrees_fahrenheit = 0
        for index, line_intersect in enumerate(intersections):
            previous_degrees_fahrenheit = degrees_fahrenheit
            degrees_fahrenheit = degrees_fahrenheit_from_points(
                needle_point=line_intersect,
                circle_center=(x, y),
                verbose=verbose
            )
            if degrees_fahrenheit < (MIN_TEMP - 15) or degrees_fahrenheit > (MAX_TEMP + 15):
                degrees_fahrenheit = previous_degrees_fahrenheit
                continue
            cv2.circle(self.image, line_intersect, 30, (255, 255, 255), 10)
            cv2.putText(
                self.image,
                f"Temp: {degrees_fahrenheit}",
                (0, height - ((index + 1) * offset)),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 255, 255),
                10
            )

        cv2.circle(self.image, (x, y), radius, (255, 255, 255), 10)
        cv2.circle(self.image, (x, y), 10, (255, 255, 255), 10)
        cv2.putText(
            self.image,
            f"Circle Size: {radius}, Center: {x}, {y}",
            (0, height),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 255, 255),
            10
        )
        if self.verbose:
            cv2.imshow('Success!', self.image)
            cv2.waitKey(0)
        cv2.imwrite(output_name, self.image)
        if degrees_fahrenheit == 0:
            raise ValueError("Could Not Get A Reading :(")
        else:
            sys.stdout.write(f"{degrees_fahrenheit}\n")
