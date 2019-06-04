import cv2
import numpy as np
from . import util

WIDTH = 224
HEIGHT = 224
PATCH_SIZE = (48, 48)

def preprocess(img):
    if not is_greyscale(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (HEIGHT, WIDTH))
    img = util.normalize(img)

    return img

def is_greyscale(img):
    return len(img.shape) < 3
