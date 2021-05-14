#!/usr/bin/env python3
"""
This script seeks to use the Hough Transform techniques to detect edges in the
images and then calculate angles and decide the temperature of the grill.
(python library)[https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html]

Fight me.
"""
import json
import os
import sys
from numpy import pi
from numpy import cos
from numpy import sin
from numpy import tan
from numpy import arctan as atan
from textwrap import dedent

import cv2
import numpy

LINES_TO_RENDER = 2
VARIANCE_IN_DEGREES = 4
VARIANCE = VARIANCE_IN_DEGREES * (pi / 180)
IGNORE_MIN_ANGLE_RADIANS = (90 - VARIANCE_IN_DEGREES) * (pi / 180)
IGNORE_MAX_ANGLE_RADIANS = (90 + VARIANCE_IN_DEGREES) * (pi / 180)
TEXT_HEIGHT = 100

# In degrees here not radians
MIN_ANGLE = 30
MAX_ANGLE = -210

# This can be fahrenheit or celsius or whatever the gauge is measuring as long
# as it is linearly measured on the gauge.
MIN_TEMP = 150
MAX_TEMP = 600


def main(image_name: str, output_name: str = None, verbose: bool = False) -> int:
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
    if verbose:
        print(f"Reading image {args.image_name}...")
    original_img = cv2.imread(image_name, 0)
    height, width = original_img.shape[:2]
    if verbose:
        print(f"Image read! It is H: {height}, W: {width}")

    if verbose:
        print("Doctoring it...")
    img = doctor_image(original_img, 71)
    if verbose:
        print("Doctoring it...")
        cv2.imshow('all better', img)
        cv2.waitKey(0)

    if verbose:
        print("Identifying circles...")
    x, y, radius = get_best_circle(img, verbose)

    if verbose:
        print("Masking out circle...")
    img = mask_circle(original_img, (x, y), radius)
    img = doctor_image(img, 31)
    if verbose:
        cv2.imshow('masquerade', img)
        cv2.waitKey(0)

    if verbose:
        print("Identifying lines...")
    points = []
    skipped_lines = 0
    for rho, theta in get_best_lines(img, verbose):
        p1, p2 = get_line_points(rho, theta, (width, height))
        angle = theta_to_degrees(theta)
        # If this is not possibly a part of the needle ignore it
        if rho > (y + 100) and (IGNORE_MIN_ANGLE_RADIANS <= theta <= IGNORE_MAX_ANGLE_RADIANS):
            skipped_lines += 1
            if verbose:
                print(f"not the dial: rho {rho} theta {theta}")
            continue
        # If this line already exists ignore it
        elif any([(p[4] - VARIANCE) <= theta <= (p[4] + VARIANCE) for p in points]):
            skipped_lines += 1
            if verbose:
                print(f"same line: rho {rho} theta {theta}")
            continue
        else:
            if verbose:
                print(f"Good line!: rho {rho} theta {theta}")
            points.append((p1, p2, angle, rho, theta))
        if len(points) >= LINES_TO_RENDER:
            break
    if verbose:
        print(f"Skipped {skipped_lines} line{'s' if skipped_lines != 1 else ''}")

    for p1, p2, angle, rho, theta in points:
        cv2.line(original_img, p1, p2, (255, 255, 255), 10)
        cv2.putText(
            original_img,
            f"Line Angle: {angle:3.2f}",
            p2,
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 255, 255),
            10
        )
        cv2.putText(
            original_img,
            f"Rho: {rho}",
            (p2[0], p2[1] + TEXT_HEIGHT),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 255, 255),
            10
        )
        cv2.putText(
            original_img,
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
        cv2.circle(original_img, line_intersect, 30, (255, 255, 255), 10)
        cv2.putText(
            original_img,
            f"Temp: {degrees_fahrenheit}",
            (0, height - ((index + 1) * offset)),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 255, 255),
            10
        )

    cv2.circle(original_img, (x, y), radius, (255, 255, 255), 10)
    cv2.circle(original_img, (x, y), 10, (255, 255, 255), 10)
    cv2.putText(
        original_img,
        f"Circle Size: {radius}, Center: {x}, {y}",
        (0, height),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (255, 255, 255),
        10
    )
    if verbose:
        cv2.imshow('Success!', original_img)
        cv2.waitKey(0)
    cv2.imwrite(output_name, original_img)
    if degrees_fahrenheit == 0:
        raise ValueError("Could Not Get A Reading :(")
    else:
        sys.stdout.write(f"{degrees_fahrenheit}\n")


def get_best_lines(img: object, verbose: bool = False) -> tuple:
    """
    Description
    -----------
    Get the rho and theta values for the lines that are found in the image.

    Params
    ------
    :image: object
    The OpenCV2 Image object stored in memory as a numpy.array.

    :verbose: bool = False
    Set to True to see this stage of the image processing and the help printed.

    Return
    ------
    Generator -> tuple
    (rho: int, theta: float)
    A generator of tuples of the rho and theta values for the lines
    found relative to the upper left corner of the picture.
    """
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    if verbose:
        cv2.imshow('uncanny', edges)
        cv2.waitKey(0)

    lines = cv2.HoughLines(edges, 1, numpy.pi / 180, 200)

    if verbose:
        print(f"There are {len(lines)} lines")
    for line in lines:
        rho, theta = line[0]
        yield int(rho), float(theta)


# TODO: Define min threshold ratio for circle size detection based on image
#       size. E.g. circle should take up no more than half the picture and
#       no less than a third
def get_best_circle(img: object, verbose: bool = False) -> tuple:
    """
    Description
    -----------
    Find the best circular object present in a given image and return a tuple
    containing the center's X, Y coordinates and the radius.

    Params
    ------
    :img: object
    The OpenCV image object stored in memory and ready to have the hough
    circle transformation applied.

    :verbose: bool = False
    Set to True to see this stage of the image processing and the help printed.

    Return
    ------
    tuple
    ((X: int, Y: int), radius: int)
    The x and y cooridnates corresponding to the pixel location of the
    circle's center as a tuple and then the radius as an int.
    """
    # Assuming that the circle is at largest taking up the whole screen and at
    # smallest, half the screen size.
    min_pixel_size = min(img.shape[:2])

    min_radius = min_pixel_size // 4
    max_radius = min_pixel_size // 2

    circles = cv2.HoughCircles(
        image=img,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    circles = numpy.uint16(numpy.around(circles))

    if verbose:
        print(f"There are {len(circles)} circles")

    # Just taking the first one
    X, Y, radius = circles[0, 0]

    return int(X), int(Y), int(radius)


def mask_circle(image, circle_center_point, radius) -> numpy.array:
    """Mask out the circle!"""
    circle_img = numpy.zeros(image.shape[:2], numpy.uint8)
    cv2.circle(circle_img, circle_center_point, radius, 255, thickness=-1)
    masked_data = cv2.bitwise_and(image, image, mask=circle_img)
    return masked_data


def get_line_points(rho: int, theta: float, screen_dimensions: tuple) -> tuple:
    """
    Description
    -----------
    Gimme dat rho and theta and you can have a line!

    Well, i mean you would have 2 points on the line where it will intercept
    the x and y axes. However, some lines intercepts are not in the first
    quadrant so I will have to adjust for that. That is why you need to give
    me the screen dimensions as well.

    Params
    ------
    :rho: int
    The rho value of the line in question.

    :theta: float
    The theta value of the line in question.

    :screen_dimensions: tuple (width: int, height: int)
    The dimensions of the screen, so I can know If the points I return are off
    of the screen and recalculate accordingly.

    Return
    ------
    tuple
    ((x1: int, y1: int), (x2: int, y2: int))
    The tuple of points which will draw the line on screen all of the way 
    across the screen.
    """
    # At pi / 2 cos(theta) = 0 making rho / cos(theta) undefined. This is the
    # case of a horizontal line making the x constant and the y values a
    # function of the intercepts on the screen border at that constant x value
    if theta == pi / 2:
        x1 = 0
        y1 = rho
        x2 = screen_dimensions[0]
        y2 = rho
    # same as the above comment, but sin(theta) = 0 on 0, pi, and 2 * pi and
    # in this case it is a vertical line not a horizontal one.
    elif theta % pi == 0:
        x1 = rho
        y1 = 0
        x2 = rho
        y2 = screen_dimensions[1]
    else:
        x1 = 0
        y1 = rho / sin(theta)
        x2 = rho / cos(theta)
        y2 = 0

    points = (int(x1), int(y1)), (int(x2), int(y2))
    return points


def theta_to_degrees(theta: float) -> int:
    """
    Description
    -----------
    Takes the angle of the detected line and
    outputs the corresponding angle in degrees.

    There is a rotation and a flip required because the computer identifies
    it from the top left corner of the screen and uses radians. I speak
    degrees, radians are confusing and I also see the image from the bottom up
    not from the top left corner down, so this translation helps me see that.

    This will bind the angle somewhere between 0 and 180 degrees. The lines do
    not have a direction so if the angle goes beyond 180 to let's say 190,
    from the computer's perspective, 190 and 10 are the same so it will just
    say 10.

    Params
    ------
    :theta: float
    The theta of the angle for the line in radians.

    Return
    ------
    int
    The angle in degrees from the observers perspective.
    """
    bounded_theta = theta % pi
    pi_over_two = (pi / 2)
    if bounded_theta == pi_over_two:
        return 0
    elif bounded_theta < pi_over_two:
        rotated_theta = pi_over_two - bounded_theta
        to_degrees = rotated_theta * (180 / pi)
        return to_degrees
    else:
        rotated_theta = pi_over_two + pi - bounded_theta
        to_degrees = rotated_theta * (180 / pi)
        return to_degrees


def doctor_image(image: object, blur_factor: int) -> numpy.array:
    """
    Description
    -----------
    Intake an OpenCV image object and output a blurred
    and grey scale version of it.

    Params
    ------
    :image: numpy.array
    The numpy array that represents an OpenCV image you wish to doctor.

    :blur_factor: int
    The factor of blur to apply to the image.
    This int must be odd and must be less than 128.

    Return
    ------
    numpy.array
    Returns the numpy array that represents the OpenCV
    image object held in memory.
    """
    img = cv2.medianBlur(image, blur_factor)
    return img


def get_line_equation(point_a: tuple, point_b: tuple) -> tuple:
    """
    Description
    -----------
    Given two points, return the slope and y-intercept for those points.
    Based on the equation:
    y = m * x + i

    Params
    ------
    :point_a: tuple
    (x, y)
    One point appearing on the line.

    :point_b: tuple
    (x, y)
    Another point appearing on the line.

    Return
    ------
    tuple
    (m: float, b: float)
    The (slope, intercept) pair for the line.
    """
    if point_a[0] == point_b[0]:
        return None, None
    else:
        m = (point_a[1] - point_b[1]) / (point_a[0] - point_b[0])
        i = point_a[1] - m * point_a[0]
        return m, i


def get_intersect_points(line_one_points: tuple, line_two_points: tuple) -> tuple:
    """
    Description
    -----------
    Given two coordinate pairs, return a point of intersection.

    Calculates things based on some middle school algebra principles of
        y = m * x + i
    to get the equation for eachline and output from that equation the
    intersection. If the two lines are parallel it will return None.

    Params
    ------
    :line_one_points: tuple
    ((x1: int, y1: int), (x2: int, y2: int))
    The tuple corresponding to two different points
    that occur on the first line.

    :line_two_points: tuple
    ((x1: int, y1: int), (x2: int, y2: int))
    The tuple corresponding to two different points
    that occur on the second line.

    Return
    ------
    tuple
    (x: int, y: int)
    The point at which these points intersect!
    If they are parallel it returns None, None.
    """
    m1, i1 = get_line_equation(*line_one_points)
    m2, i2 = get_line_equation(*line_two_points)

    # In the case either of the lines are vertical or both are parallel.
    if m1 == m2:
        return None, None
    elif m1 is None:
        x_intersect = line_one_points[0]
        y_intersect = m2 * x_intersect + i2
        return int(x_intersect), int(y_intersect)
    elif m2 is None:
        x_intersect = line_two_points[0]
        y_intersect = m1 * x_intersect + i1
        return int(x_intersect), int(y_intersect)

    # Otherwise if they are not parallel or vertical
    else:
        x_intersect = (i2 - i1)/(m1 - m2)
        y_intersect = m1 * x_intersect + i1
        return int(x_intersect), int(y_intersect)


def get_circle_intersects(center: tuple, radius: int, rho: int, theta: float) -> tuple:
    """
    Description
    -----------
    This will intake the line's rho and theta as well as the circle's center
    and radius and return the tuple of 2 (x, y) cooridnate pairs corresponding
    to the intersects on the circle's perimeter with the line. Whether or not
    the line actually intersects the circle, this equation will align the line
    and the circle to the origin off the image (0, 0) calculate the intersects
    based solely on the angle and radius, then return the points offset by the
    circle's center point

    Params
    ------
    :center: tuple
    (x: int, y: int)
    The center point of the circle.

    :radius: int
    The radius of the circle.

    :rho: int
    The rho value of the line.

    :theta: float
    The theta value of the line.

    Return
    ------
    tuple
    ((x: int, y: int), (x: int, y: int))
    The pixel coordinates for the points on the edge of the circle where the
    line intersects with the circle.
    """
    # Because this is how I wrote the math with center = (h, k).
    h = center[0]
    k = center[1]
    # I will actually translate it so that the circle is centered on the
    # origin and so is the line, making the equation soooo much easier
    # to calculate.

    # tan(theta) is undefined at pi / 2 so in that case we return early
    # because the line is vertical and the calculations below break.
    if theta == pi / 2:
        # In this case we just assume it cuts through the center of the circle
        # and comes out at the top and bottom exactly the radius distance from
        # the center.
        point_a = (int(h), int(k + radius))
        point_b = (int(h), int(k - radius))
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
        return ((int(x1), int(y1)), (int(x2), int(y2)))


def degrees_fahrenheit_from_points(needle_point: tuple, circle_center: tuple, verbose: bool = False) -> int:
    """
    Description
    -----------
    Calculate the degrees fahrenheit for this particular gauge given the point
    and the circle's center.

    Params
    ------
    :needle_point: tuple
    The coordinates for the point that correlates to where the gauge needle is
    pointing.

    :circle_center: tuple
    The coordinates for the center of the circle.

    :verbose: bool = False
    Set to True to see this stage of the image processing and the help printed.

    Return
    ------
    int
    The degrees fahrenheit calculated based on the data provided.
    """
    needle_x = needle_point[0]
    needle_y = needle_point[1]
    circle_x = circle_center[0]
    circle_y = circle_center[1]


    # Quadrants are like so:
    #  /--+--\
    # | 2 | 1 |
    # +---+---+
    # | 3 | 4 |
    #  \--+--/
    # But from the perspective of the screen's top left, so really just upside
    # down from what you're seeing.
    if needle_x > circle_x:
        if needle_y >= circle_y:
            quadrant = 2
        elif needle_y < circle_y:
            quadrant = 3
    elif needle_x < circle_x:
        if needle_y >= circle_y:
            quadrant = 1
        elif needle_y < circle_y:
            quadrant = 4
    elif needle_x == circle_x:
        if needle_y >= circle_y:
            quadrant = 1
        elif needle_y < circle_y:
            quadrant = 4
        elif needle_y == circle_y:
            raise ValueError("Those are the same point.")

    raw_angle = -atan(
        (circle_y - needle_y)
        /
        (circle_x - needle_x)
    )
    # if verbose:
    if verbose:
        print(f"The raw angle: {raw_angle * (180 / pi)}")
        print(f"The quadrant : {quadrant}")

    angle_adjustment = {
        1: 0,
        2: -180,
        3: -180,
        4: 0
    }
    adjusted_angle = raw_angle * (180 / pi) + angle_adjustment[quadrant]
    if verbose:
        print(f"Adjusted angle: {adjusted_angle}")

    # Using the y = m * x + b formula for the equation for a line to calculate
    # the linear relationship between the angle in degrees and the temperature
    # on the grill.
    b = 210
    m = (MAX_TEMP - MIN_TEMP)  / (MAX_ANGLE - MIN_ANGLE)
    temp_calc = m * adjusted_angle + b
    if verbose:
        print(f"Maybe this is the temp: {temp_calc:.2f}?")
    return int(temp_calc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Give me an image and I will give you some data...")
    parser.add_argument(
        "image_name",
        help="The image to read and process."
    )
    parser.add_argument(
        "output_name",
        help="The location where the output file will be written to."
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Whether or not to print help and show images along to way or just to process quickly."
    )
    args = parser.parse_args()

    main(
        image_name=args.image_name,
        output_name=args.output_name,
        verbose=args.verbose
    )
