#!/usr/bin/env python3
"""
Description
-----------
This module contains code to store some trigonometry/geometry/algebra
functions that I created which are useful in solving this set of problems.
"""
from numpy import pi

from . import constants


def determine_angle_between_points(point_a: tuple, point_b: tuple) -> int:
    """
    Description
    -----------
    The angle in between the two points in degrees, point A is the center.

    Params
    ------
    :point_a: tuple
    The (X, Y) cooridnates for the center point of the two points.

    :point_b: tuple
    The (X, Y) cooridnates for the orbiting point of the two points.

    Return
    ------
    int
    The angle in between the two points.
    """


def theta_to_degrees(theta: float) -> float:
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
    if bounded_theta == constants.PI_OVER_TWO:
        return 0
    elif bounded_theta < constants.PI_OVER_TWO:
        rotated_theta = constants.PI_OVER_TWO - bounded_theta
        to_degrees = rotated_theta * constants.RADS_TO_DEGS
        return to_degrees
    else:
        rotated_theta = constants.PI_OVER_TWO + pi - bounded_theta
        to_degrees = rotated_theta * constants.RADS_TO_DEGS
        return to_degrees


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
