"""
written by James Go
"""

import numpy as np
import sys
import os
from os.path import join as opj
from PIL import Image
from mod.util import extract_from_mask

def main():
    """
    this script assumes each image has a mask and each mask has an image and they all have the same name
    """
    image_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    out_dir = sys.argv[3]

    image_files = get_all_files(image_dir)
    mask_files = get_all_files(image_dir)
    if len(set(image_files) & set(mask_files)) != len(image_files):
        raise Exception("directories don't have matching files")

    for image_file in image_files:
        mask = Image.open(opj(mask_dir, image_file))
        image = Image.open(opj(image_dir, image_file))
        new_image = Image.fromarray(extract_from_mask(np.array(image), np.array(mask)))
        new_image.save(opj(out_dir, image_file))

def get_all_files(d):
    return [f for f in os.listdir(d) if os.path.isfile(opj(d, f))]

main()
