import sys
import cv2
import numpy as np
import os
from os.path import join as opj
from mod.model import Segmenter
import tensorflow as tf
from PIL import Image
from mod import preprocess

def main():
    model_file = sys.argv[1]
    image_dir = sys.argv[2]
    out_dir = sys.argv[3]

    graph = tf.get_default_graph()
    segmenter = Segmenter(model_file, graph)

    filenames = os.listdir(image_dir)
    for fn in filenames:
        img = Image.open(opj(image_dir, fn))
        #img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = preprocess.preprocess(np.array(img))
        img = segmenter.segmenter(img)
        img = Image.fromarray(np.uint8(img * 255))
        img = img.convert("L")
        img.save(opj(out_dir, fn))

main()
