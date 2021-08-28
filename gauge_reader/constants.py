#!/usr/bin/env python3
"""
Description
-----------
This module simply stores some useful constant variables for reuse later.
"""
from numpy import pi

RADS_TO_DEGS = (180 / pi)
DEGS_TO_RADS = (pi / 180)
PI_OVER_TWO = pi / 2

LINES_TO_RENDER = 2
VARIANCE_IN_DEGREES = 4
VARIANCE = VARIANCE_IN_DEGREES * (pi / 180)
IGNORE_MIN_ANGLE_RADIANS = (90 - VARIANCE_IN_DEGREES) * (pi / 180)
IGNORE_MAX_ANGLE_RADIANS = (90 + VARIANCE_IN_DEGREES) * (pi / 180)
DISTANCE_THRESHOLD = 100
TEXT_HEIGHT = 100

# In degrees here not radians
MIN_ANGLE = 30
MAX_ANGLE = -210

# This can be fahrenheit or celsius or whatever the gauge is measuring as long
# as it is linearly measured on the gauge.
MIN_TEMP = 150
MAX_TEMP = 600
