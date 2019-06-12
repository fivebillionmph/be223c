import csv
import sys
from mod import model
from mod import preprocess
import os
from PIL import Image
from os.path import join as opj
import tensorflow as tf
import numpy as np
from mod import util

def main():
    test_labels = sys.argv[1]
    image_dir = sys.argv[2]
    lesion_mask_dir = sys.argv[3]
    out_dir = sys.argv[4]
    model_file = sys.argv[5]
    seg_model_file = sys.argv[6]

    graph = tf.get_default_graph()
    segmenter = model.Segmenter(seg_model_file, graph)
    classifier = model.Classifier2(model_file, graph)

    file_data = []
    with open(test_labels) as f:
        reader = csv.reader(f)
        header = next(reader)
        header.append("Prediction")

        for line in reader:
            fn = line[0]
            img = Image.open(opj(image_dir, fn + ".png"))
            lesion_mask = Image.open(opj(lesion_mask_dir, fn + ".png"))
            img = preprocess.preprocess(np.array(img))
            img = segmenter.segmenter(img)
            patch_coords = util.find_lesion_coordiates(np.array(lesion_mask))
            patch = util.extract_patch(np.array(img), patch_coords, preprocess.PATCH_SIZE)
            prediction = classifier.classify(patch)
            line.append(prediction)
            file_data.append(line)

    with open(opj(out_dir, "results.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for line in file_data:
            writer.writerow(line)

main()
