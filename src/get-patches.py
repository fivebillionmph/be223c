"""
written by James Go
"""

from mod import util, preprocess
import sys
import os
from os.path import join as opj
from os.path import splitext as ops
import PIL.ImageOps
from PIL import Image
import numpy as np
import cv2

def main():
    """
    this script extracts patches using images and lesion masks
    images with the same name will be matched between the two input directories

    CLI Args:
        1: a directory of segmented lung images
        2: a directory of lesion masks
        3: the output directory for the patches
    """
    image_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    out_dir = sys.argv[3]

    image_files = get_all_files(image_dir)
    mask_files = get_all_files(mask_dir)

    check_files = list(set([ops(f)[0] for f in image_files]) & set([ops(f)[0] for f in mask_files]))

    for cf in check_files:
        mask_file = [f for f in mask_files if cf == ops(f)[0]][0]
        image_file = [f for f in image_files if cf == ops(f)[0]][0]
        mask = Image.open(opj(mask_dir, mask_file))
        #mask = PIL.ImageOps.invert(mask)
        patch_coords = util.find_lesion_coordiates(np.array(mask))
        image = Image.open(opj(image_dir, image_file))
        #image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        #image = preprocess.preprocess(image)
        #print(cf, patch_coords, image.shape)
        patch = util.extract_patch(np.array(image), patch_coords, preprocess.PATCH_SIZE)
        patch_image = Image.fromarray(np.uint8(patch)).convert("L")
        patch_image.save(opj(out_dir, cf + ".png"), "PNG")
        #image = Image.fromarray(np.array(image * 255)).convert("L")
        #image.save(opj(out_dir, cf + ".png"), "PNG")

def get_all_files(d):
    return [f for f in os.listdir(d) if os.path.isfile(opj(d, f))]

main()
