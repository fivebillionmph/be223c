import cv2
import numpy as np

WIDTH = 224
HEIGHT = 224
PATCH_SIZE = (48, 48)

def normalize(img):
    # Normalize the images in the range of 0 to 1 (converted into float64)

    return (img-np.min(img))/(np.max(img)-np.min(img))
    # return img / np.max(img)

def preprocess(img):
    if not is_greyscale(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (HEIGHT, WIDTH))
    img = normalize(img)

    return img

def is_greyscale(img):
    return len(img.shape) < 3
