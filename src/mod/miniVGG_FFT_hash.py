# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 07:50:27 2019

@author: josep
"""

""" Mini-VGG FFT Hash """

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
    def __init__(self, images_dir, processing_func, model_file, graph):
        """for each image: hash = hashing_func(processing_func(image))
           save hash in list """

        # initialize the number of epochs to train for, base learning rate, and batch size
        self.images = []
        self.hashes = []
        self.mean_activations = []
        self.graph = graph
        self.activation_model = keras.models.load_model(model_file)
        self.image_names = os.listdir(images_dir)
        self.processing_func = processing_func
        
        for i in range(len(self.image_names)):
            path = opj(images_dir, self.image_names[i])
            
            im = Image.open(path)
            im = processing_func(np.array(im))
            im = cv2.resize(im, (128,128))
            im = im / 255.0
            self.images.append(im)
        
        # Returns a list of five Numpy arrays: one array per layer activation
        images_2 = np.expand_dims(np.array(self.images), 3)

        with self.graph.as_default():
            activations = self.activation_model.predict(images_2)
        pool = np.array(activations[-9])
        
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
            find most similar images by hamming distance or some other metric """

        #image_orig = self.processing_func(np.array(image))
        image_orig = cv2.resize(image, (128,128))
        image = image_orig / 255.0
        #print(image.shape)

        matches = []
        hammings = []
        
        # Returns a list of five Numpy arrays: one array per layer activation
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)
        
        with self.graph.as_default():
            activations = self.activation_model.predict(image)

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
        
        for i in range(len(self.hashes)):
            hammings.append(int(hamming(query_hash, self.hashes[i]) * 128))
            
            # https://stackoverflow.com/questions/22736641/xor-on-two-lists-in-python
        
        for i in range(len(hammings)):
            if hammings[i] <= 40:
                matches.append({"name": self.image_names[i], "similarity": hammings[i]})

        return matches

def miniVGG(width, height, depth, classes):
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
