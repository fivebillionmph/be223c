"""
author: Mohammadali Alidoost
this code gets only the images to trian the pre-trained model using the cross validation technique
"""

############################################################################################################################################################################################
# import all necessary libraries

import os
# to run on GPU, comment the following tow lines 
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import pylab
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from util_pre import read_data
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from keras.layers import Input, Flatten, Dense, Dropout, Convolution2D, Conv2D, MaxPooling2D, Lambda, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, AveragePooling2D, Concatenate
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import random
############################################################################################################################################################################################
# get all of the input parameters and paths to train and tune

# input parameters and paths for training

csvPath = '/home/mohammadali/Downloads/Run/Train.csv'
imagePath = '/home/mohammadali/Downloads/Run/Train-Seg-Man'
inputSize = (224,224)
IMG_SIZE = 224

k_fold = 2
batch_size = 32
learning_rate = 1e-6
epoch_size = 2
nodes = 1024 # number of nodes for the fc layer
drop_out = 0.5 # the layer after the fc layer

# input parameters and paths for testing

test_dir = '/home/mohammadali/Downloads/Run/Test-Seg-Man'
csvTest = '/home/mohammadali/Downloads/Run/Test.csv'
############################################################################################################################################################################################
def main():
    """ train the model """

    # preparing the data to train

    random.seed(7)

    train_img,train_label,val_img,val_label = read_data(csvPath,
        imagePath, inputSize, 0.00001, split_by_id=False, normalize=True, crop_image=True)
    train_img = np.stack( (train_img[:,:,:,0],)*3, axis=-1 )
    X = train_img
    y = train_label

    print("")
    print("using images")
    print("")
    print("train_images:", train_img.shape)
    print("train_labels:", train_label.shape)
    #print("validation_images:", val_img.shape)
    #print("validation_label:", val_label.shape)

    # using the pretrained model for training

    print("")
    print("getting the model")
    print("")
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #conv_base.summary()

    for layer in conv_base.layers[:-4]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    #for layer in conv_base.layers:
    #    print(layer, layer.trainable)

    # making the FC layers of the model

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(nodes, activation='relu'))
    model.add(layers.Dropout(drop_out))
    model.add(layers.Dense(1, activation='sigmoid'))
    #model.summary()

    # compiling the model

    model.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=learning_rate), metrics=['acc'])

    # defining k-fold cross validation test harness

    kfold = StratifiedKFold(n_splits = k_fold, shuffle = True, random_state = np.random.seed(7))
    cvscores1 = []
    cvscores2 = []
    i = 1
    for train, test in kfold.split(X, y):
    
        # data augmentation

        train_datagen = ImageDataGenerator(rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')

        #val_datagen = ImageDataGenerator(rescale=1./255)

        # training the model

        print('fold %d:' % (i))
        H = model.fit_generator(
            train_datagen.flow(X[train], y[train], batch_size = batch_size),
            steps_per_epoch=len(X[train]) // batch_size,
            epochs=epoch_size,
            validation_data=(X[test], y[test]),
            validation_steps=len(X[test]) // batch_size)

        # computing the accuracy metric for this CV fold
    
        scores = model.evaluate(X[test], y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores1.append(scores[1] * 100)
    
        # computing the AUC metric for this CV fold

        preds = model.predict(X[test])
        fpr, tpr, thresholds = metrics.roc_curve(y[test], preds)
        roc_auc = metrics.auc(fpr, tpr)
        print("%s: %.2f%%" % ('auc', roc_auc*100));
        cvscores2.append(roc_auc*100)
        i = i + 1
        print('');

    print("tot_acc_avg: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores1), np.std(cvscores1)));
    print("tot_auc_avg: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores2), np.std(cvscores2)));

    # saving the model

    model.save_weights('model_weights_lung_pro_cv_image.h5')
    model.save('model_lung_pro_cv_image.h5')

    # test the model

    # loading the model

    #model = load_model('model_lung_pro_cv_image.h5')
    #model.summary()

    # getting the images for testing

    file_list2 = os.listdir(test_dir)
    test_imgs = [test_dir + "/" + "{}".format(i) for i in file_list2]
    #print("No. of test images = ", len(test_imgs))
    X_test = []
    for image in test_imgs:
        X_test.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_CUBIC))

    X_test = np.array(X_test)
    X_test = (X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))
    #X_test = X_test/255.0
    print("")
    print("shape of X_test:", X_test.shape)

    df = pandas.read_csv(csvTest)
    #print('shape of the dataframe:', df.shape)
    #print(df.head(2))

    # getting the true labels for testing

    na = df.loc[:,'File']
    la = df.loc[:,'Progression']
    na = np.array(na)
    la = np.array(la)
    I = np.argsort(na)
    na = na[I]
    la = la[I]
    y_test = la
    #sns.set(rc={'figure.figsize':(5,4)})
    #sns.countplot (y_test)
    #plt.title("Labels")
    print ("shape of y_test:", y_test.shape)

    # model prediction

    preds_test = model.predict(X_test, verbose=1)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)
    predictions_test = preds_test_t [:, 0]
    print("")
    print("Predicted labels:", predictions_test)
    print("")
    print("True labels:", y_test)
    print("")
    com = np.isclose(predictions_test, y_test.T)
    print (com)
    #true_prediction_number = 1 * (com == 'True')
    #print(true_prediction_number)

    print("")
    fpr, tpr, thresholds = metrics.roc_curve(y_test, preds_test)
    roc_auc = metrics.auc(fpr, tpr)
    print("%s: %.2f%%" % ('auc', roc_auc*100))

    cm = confusion_matrix(y_test, predictions_test)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return

if __name__ == "__main__":
    csvPath = csvPath
    imagePath = imagePath
    inputSize = inputSize
    IMG_SIZE = IMG_SIZE
    k_fold = k_fold
    batch_size = batch_size
    learning_rate = learning_rate
    epoch_size = epoch_size
    nodes = nodes
    drop_out = drop_out
    test_dir = test_dir
    csvTest = csvTest

    main()

