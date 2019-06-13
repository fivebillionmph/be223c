"""
Author: Joseph Tsung
Mini-VGG FFT Hash
"""
# import the necessary packages
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
from os.path import join as opj
import cv2

from scipy.spatial.distance import hamming
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import jaccard_similarity_score, mutual_info_score

from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import SGD
from keras.utils import np_utils
import keras

class ImageSimilarity:
    """ Instantiate a class for storing images and hashes """
    
    def __init__(self, images_dir, processing_func, model_file, graph):
        """for each image: hash = hashing_func(processing_func(image))
           save hash in list
        
        Parameters:
        -path or directory to the image folder
        -preprocessing function for the images
        """

        # initialize the number of epochs to train for, base learning rate, and batch size
        self.images = []
        self.hashes = []
        self.mean_activations = []
        self.graph = graph
        self.activation_model = keras.models.load_model(model_file)
        self.image_names = os.listdir(images_dir)
        self.processing_func = processing_func
        
        # loop through image directory to store images, paths
        for i in range(len(self.image_names)):
            path = opj(images_dir, self.image_names[i])
            
            im = Image.open(path)
            im = cv2.resize(np.array(im), (128,128))
            im = processing_func(im)
            # im = im / 255.0
            self.images.append(im)
        
        # Returns image numpy array to feed into the model
        images_2 = np.expand_dims(np.array(self.images), 3)

        # get all activations from model layers
        with self.graph.as_default():
            activations = self.activation_model.predict(images_2)
        
        # pick out the output from the pooling layer of interest; output size (32x32x128)
        pool = np.array(activations[-9])
        
        # calculate mean activations for this pooling layer for each of the 128 feature maps
        # pool.shape[0] tells you how many pictures we're trying to hash
        # pool.shape[3] is 128 for each of the 128 feature maps
        for i in range(pool.shape[0]):
            means = []

            for j in range(pool.shape[3]):
        
                temp = pool[i,:,:,j]
                
                mean = temp.mean()
                means.append(mean)
                
            self.mean_activations.append(means)
        
        # Materials from: https://github.com/gabrielpierobon/cnnshapes/blob/master/README.md
        
        for i in range(len(self.mean_activations)):

            h = []

            # apply FFT to the activations
            fourier = np.fft.fft(self.mean_activations[i])

            # normalize by DC component
            fourier = fourier / fourier[0]

            # get only the real parts
            real = np.real(fourier)

            # binarize
            for j in range(len(real)):
                if real[j] > 0:
                    h.append(1)
                else:
                    h.append(0)
            self.hashes.append(h)

    def query_image(self, image):
        """ image_hash = hashing_func(processing_func(image))
            find most similar images by hamming distance 
        
        Parameters:
        -query image
        
        Output:
        -final_matches: paths to the similar images
        """

        #image_orig = self.processing_func(np.array(image))
        image_orig = image * 255
        image_orig = cv2.resize(image_orig, (128,128))
        image = image_orig / 255
        image_orig = image_orig.astype(dtype=np.uint8)
        #print(image.shape)

        matches = []
        hammings = []
        
        # Returns image numpy array to feed into the model
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)
        
        # Get activations for the query image
        with self.graph.as_default():
            activations = self.activation_model.predict(image)
        
        # Get activations from the pooling layer (32x32x128)
        pool = np.array(activations[-9])

        for i in range(pool.shape[0]):

            image_activations = []
            means = []
            
            for j in range(pool.shape[3]):
        
                temp = pool[i,:,:,j]
                
                mean = temp.mean()
                means.append(mean)
                
            image_activations.append(means)
        
        for i in range(len(image_activations)):
            
            h = []
            
            # apply FFT to the activations
            fourier = np.fft.fft(image_activations[i])
        
            # normalize by DC component
            fourier = fourier / fourier[0]
        
            # get only the real parts
            real = np.real(fourier)
        
            # binarize
            for j in range(len(real)):
                if real[j] > 0:
                    h.append(1)
                else:
                    h.append(0)
            query_hash = h
        
        # Generate hamming distance (i.e. how many out of the 128-bits are different)
        for i in range(len(self.hashes)):
            hammings.append(int(hamming(query_hash, self.hashes[i]) * 128))
            
            # https://stackoverflow.com/questions/22736641/xor-on-two-lists-in-python
        
        for i in range(len(hammings)):
            if hammings[i] <= 5:
                #im = np.where(self.images[i].flatten() > 0, 1, 0)
                #image_orig = np.where(image_orig.flatten() > 0, 1, 0)
                #mi = mutual_info_score(im, image_orig)
                #jac = jaccard_similarity_score(im, image_orig)
                im = self.images[i] * 255
                mi = mutual_info_score(im.astype(dtype=np.uint8).flatten(), image_orig.flatten())
                jac = jaccard_similarity_score(im.astype(dtype=np.uint8).flatten(), image_orig.flatten())
                matches.append({"name": self.image_names[i], "similarity": hammings[i], "mutual_info_score": mi, "jaccard_similarity_score": jac})

        return matches

def miniVGG(width, height, depth, classes):
    """ the base model CNN for the activations for hashing """
    
    inputShape = (height, width, depth)
    model = Sequential()

    # first CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    # second CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # third CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
 
    # softmax classifier
    model.add(Dense(classes-1))
    model.add(Activation("softmax"))
 
    # return the constructed network architecture
    return model

def train_model(images_dir, labels_file, processing_func):
    """ model training in order to get the activations for hashing
    
    Parameters:
    -images_dir: path or directory to the folder of images
    -labels_file: csv file containing the response/no response label for the predictions
    -processing_func: preprocessing function for the images
    
    Output:
    -activations for hashing
    """
    
    images = []

    NUM_EPOCHS = 5
    INIT_LR = 1e-4
    BS = 32
    num_classes = 2 

    df = pd.read_csv(labels_file)

    paths = df['File']
    
    for i in range(len(paths)):
        path = opj(images_dir, paths[i] + ".png")
        
        im = Image.open(path)
        im = processing_func(np.array(im))
        im = cv2.resize(im, (128,128))
        im = im / 255.0
        images.append(im)

    labels = df['Progression']
    labels = np.array(labels)

    (trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=0.10, random_state=42)
    
    trainX = np.array(trainX)
    testX = np.array(testX)
    trainY = np.array(trainY)
    testY = np.array(testY)
    
    trainX = np.expand_dims(trainX, 3)
    testX = np.expand_dims(testX, 3)

    # initialize the label names
    labelNames = ["no response", "response"]
    
    # initialize the optimizer and model
    opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
    model = miniVGG(width=128, height=128, depth=1, classes=2)
    # check for structure of the model
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    # train the network
    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=BS, epochs=NUM_EPOCHS)

    # Extracts the outputs of all layers
    layer_outputs = [layer.output for layer in model.layers]
    layer_names = [layer.name for layer in model.layers]
     
    # Creates a model that will return these outputs, given the model input
    activation_model = Model(inputs = model.input, outputs = layer_outputs) 

    return activation_model
