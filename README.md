## Project overview
### Directories

- [/data](https://github.com/fivebillionmph/be223c/tree/master/data)
  - This is where all the variable or generated data is kept.  This includes the classification models, miniVGG model, automatic lung segmentation model, model test results and the lung images (original images, lesion segmentations, patches, etc).
  - This directory is not as organized, it has unused models and images because data was swapped out and interchanged.
  - It is gitignored, but the data is packaged in the zip file submitted to CCLE.

- /data/miniVGG.h5
  - the miniVGG hash model HDF5 file

- /data/segs2/patches-training
  - directory of patches that are hashed and checked against for the content based retrieval image similarity

- /data/lesion_classification.model
  - HDF5 file for the UNET encoder based whole image lesion classification model (model 1)

- /data/model_lung_pro_cv_patch.h5
  - HDF5 file for the VGG16 based patch classification model (model 2)

- /data/model_lung_pro_cv_image1.h5
  - HDF5 file for the VGG16 based whole image classification model (model 3)

- /data/lung_seg.mode
  - HDF5 file for the lung segmentation model

- /data/Train.csv
  - the training set images and their labels

- /data/Test.csv
  - the test set images and their labels

- /data/test-model\*
  - the test results files for the 3 models which ROC, AUC, etc...

- [/scripts](https://github.com/fivebillionmph/be223c/tree/master/scripts)
  - Simple scripts for starting the web server and getting the installed Python modules

- [/src](https://github.com/fivebillionmph/be223c/tree/master/src)
  - Python scripts for running the server, segmenting, testing models, training the miniVGG, etc.

- [/src/mod](https://github.com/fivebillionmph/be223c/tree/master/src/mod)
  - Shared modules used by scripts in /src.

- [/src2](https://github.com/fivebillionmph/be223c/tree/master/src2)
  - Other scripts for generating models or segmenting.

- [/web](https://github.com/fivebillionmph/be223c/tree/master/web)
  - HTML templates and static CSS, JS and images
