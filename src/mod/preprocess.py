"""
Author: James Go

preprocessing and other small functions for handing images
"""

import cv2
import numpy as np
from . import util

WIDTH = 224
HEIGHT = 224
PATCH_SIZE = (64, 64)

def preprocess(img):
    """
    standard function for normalizing images

    Args:
        in image

    Returns:
        a greyscaled normalized image
    """
    if not is_greyscale(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = util.normalize(img)

    return img

def resize(img):
    img = cv2.resize(img, (HEIGHT, WIDTH))
    return img

def is_greyscale(img):
    return len(img.shape) < 3

def translate_patch_coordinates(img, point):
    """
    translates coordinates on the scale of (WIDTH, HEIGHT) and
    translates them up to the images real size

    this is needed because the images displayed on the website always have the same width/height,
    but it is not the same real size as the image

    Args:
        img: an image
        point: dictionary with x and y integer values

    Returns:
        a dictionary with the translated x and y values
    """
    scale_w = img.shape[0] / WIDTH
    scale_h = img.shape[1] / HEIGHT
    new_point = {
        "x": int(point["x"] * scale_w),
        "y": int(point["y"] * scale_h),
    }
    return new_point
