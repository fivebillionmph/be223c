import cv2
import numpy as np
from . import util

WIDTH = 224
HEIGHT = 224
PATCH_SIZE = (64, 64)

def preprocess(img):
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
    scale_w = img.shape[0] / WIDTH
    scale_h = img.shape[1] / HEIGHT
    new_point = {
        "x": int(point["x"] * scale_w),
        "y": int(point["y"] * scale_h),
    }
    return new_point
