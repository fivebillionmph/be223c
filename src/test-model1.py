import csv
import sys
from mod import model
from mod import preprocess
import os
from PIL import Image
from os.path import join as opj
import tensorflow as tf
import numpy as np

def main():
    test_labels = sys.argv[1]
    image_dir = sys.argv[2]
    out_dir = sys.argv[3]
    model_file = sys.argv[4]
    seg_model_file = sys.argv[5]

    graph = tf.get_default_graph()
    segmenter = model.Segmenter(seg_model_file, graph)
    classifier = model.Classifier1(model_file, graph)

    file_data = []
    with open(test_labels) as f:
        reader = csv.reader(f)
        header = next(reader)
        header.append("Prediction")

        for line in reader:
            fn = line[0]
            img = Image.open(opj(image_dir, fn + ".png"))
            img = preprocess.preprocess(np.array(img))
            img = segmenter.segmenter(img)
            img = preprocess.preprocess(np.array(img))
            prediction = classifier.classify(img)
            line.append(prediction)
            file_data.append(line)

    with open(opj(out_dir, "results.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for line in file_data:
            writer.writerow(line)

main()
