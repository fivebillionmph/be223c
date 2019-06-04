import keras
import numpy as np

class Model:
    def __init__(self, filename, graph):
        # why graph is needed:
        # https://kobkrit.com/tensor-something-is-not-an-element-of-this-graph-error-in-keras-on-flask-web-server-4173a8fe15e1
        self.model = keras.models.load_model(filename)
        self.graph = graph

    def prob(self, img):
        return 1.0
        with self.graph.as_default():
            return float(self.model.predict(np.array([img]))[0][0])
