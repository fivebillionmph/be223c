import keras
import numpy as np
import cv2
from .seg_lung import infer_seg
from .util import extract_from_mask
from .preprocess import preprocess

class Classifier1:
    def __init__(self, filename, graph):
        # why graph is needed:
        # https://kobkrit.com/tensor-something-is-not-an-element-of-this-graph-error-in-keras-on-flask-web-server-4173a8fe15e1
        self.model = keras.models.load_model(filename)
        self.graph = graph

    def classify(self, img):
        with self.graph.as_default():
            input_size = ( int(self.model.inputs[0].shape[1]), int(self.model.inputs[0].shape[2]) )
            img = cv2.resize(img, (input_size[1], input_size[0]))
            img = preprocess(img)
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)
            return float(self.model.predict(img)[0])

class Classifier2:
    def __init__(self, filename, graph):
        self.model = keras.models.load_model(filename)
        self.graph = graph

    def classify(self, img):
        with self.graph.as_default():
            input_size = ( int(self.model.inputs[0].shape[1]), int(self.model.inputs[0].shape[2]) )
            img = cv2.resize(img, (input_size[1], input_size[0]))
            img = preprocess(img)
            #img = cv2.merge((img, img, img))
            img = np.stack([img] * 3, axis = -1)
            return float(self.model.predict(np.array([img]))[0][0])

class Segmenter:
    def __init__(self, filename, graph):
        self.model = keras.models.load_model(filename)
        self.graph = graph

    def segmenter(self, img):
        with self.graph.as_default():
            mask = infer_seg(img, self.model)
        img = extract_from_mask(img, mask)
        return img
