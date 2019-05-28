import cv2
import numpy as np

WIDTH = 224
HEIGHT = 224

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def preprocess(img):
    img = cv2.resize(img, (HEIGHT, WIDTH))
    img = normalize(img)
    return img
