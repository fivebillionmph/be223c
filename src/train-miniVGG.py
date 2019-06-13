"""
Author: James Go
"""

from mod.miniVGG_FFT_hash import train_model
from mod.preprocess import preprocess
import sys

def main():
    """
    runs the miniVGG train_model function

    CLI Args:
        1: the directory of patch images
        2: the csv training label file
        3: the output model filename
    """
    model = train_model(sys.argv[1], sys.argv[2], preprocess)
    model.save(sys.argv[3])

main()
