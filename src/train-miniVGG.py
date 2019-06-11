from mod.miniVGG_FFT_hash import train_model
from mod.preprocess import preprocess
import sys

def main():
    model = train_model(sys.argv[1], sys.argv[2], preprocess)
    model.save(sys.argv[3])

main()
