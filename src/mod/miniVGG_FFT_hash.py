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
import pydicom
import pandas as pd
import os.path
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

class ImageSimilarity2:
    
    def __init__(self, images_dir, processing_func):
        """for each image: hash = hashing_func(processing_func(image))
           save hash in list """
        
        # initialize the number of epochs to train for, base learning rate, and batch size
        NUM_EPOCHS = 5
        INIT_LR = 1e-4
        BS = 32
        num_classes = 2 
        
        self.images = []
        self.hashes = []
        self.mean_activations = []
        self.model = Model()
        
        #print("%.0f pictures found in root directory" % len(os.listdir(images_dir)))
        
        df = pd.read_csv(r"C:\workspace\223C\Master3.csv")
        self.paths = df['File']
        
        for i in range(len(self.paths)):
         
            path = images_dir + "\\" + self.paths[i] + ".png"   
            
            try:
                im = Image.open(path)
                im = np.array(im)
                im = cv2.resize(im, (128,128))
                im = im / 255.0
                self.images.append(im)
                #print(path)
                
            except FileNotFoundError:
                #print("Image %s not found" % self.paths[i])
        
        labels = df['Progression']
        labels = np.array(labels)
        
        #print("[INFO] preparing training and testing datasets...")
        (trainX, testX, trainY, testY) = train_test_split(self.images, labels, test_size=0.10, random_state=42)
        
        trainX = np.array(trainX)
        testX = np.array(testX)
        trainY = np.array(trainY)
        testY = np.array(testY)
        
        trainX = np.expand_dims(trainX, 3)
        testX = np.expand_dims(testX, 3)
         
        # one-hot encode the training and testing labels
        #trainY = np_utils.to_categorical(trainY, 2)
        #testY = np_utils.to_categorical(testY, 2)
         
        # initialize the label names
        labelNames = ["no response", "response"]
        
        # initialize the optimizer and model
        #print("[INFO] compiling model...")
        opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
        self.model = miniVGG(width=128, height=128, depth=1, classes=2)
        # check for structure of the model
        #model.summary()
        self.model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        
        # train the network
        #print("[INFO] training model...")
        H = self.model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=BS, epochs=NUM_EPOCHS)
        
        # make predictions on the test set
        preds = self.model.predict(testX)
         
        # show a nicely formatted classification report
        #print("[INFO] evaluating network...")
        #print(classification_report(testY, preds, target_names=labelNames))
        
        # compute the confusion matrix and and use it to derive accuracy, sensitivity, and specificity
        #cm = confusion_matrix(testY, preds.argmax(axis=1), labels=range(num_classes))
        #total = sum(sum(cm))
        #acc = (cm[0, 0] + cm[1, 1]) / total
        #sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        #specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
            
        #loss, acc = model.evaluate(testX, testY, verbose = 1)
        #print("accuracy: %f" % acc)
        #print(cm)
        #print("sensitivity: %f" %sensitivity)
        #print("specificity: %f" %specificity)
         
        # plot the training loss and accuracy
        N = NUM_EPOCHS
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.show()
        
        ## Returns the values of the layer activations
        
        # Extracts the outputs of all layers
        self.layer_outputs = [layer.output for layer in self.model.layers]
        layer_names = [layer.name for layer in self.model.layers]
         
        # Creates a model that will return these outputs, given the model input
        self.activation_model = Model(inputs = self.model.input, outputs = self.layer_outputs) 
        
        # Returns a list of five Numpy arrays: one array per layer activation
        images_2 = np.expand_dims(np.array(self.images), 3)
        
        activations = self.activation_model.predict(images_2)
        pool = np.array(activations[-9])
        #print(pool.shape)
        
        for i in range(pool.shape[0]):
            
            means = []
            
            #print("Calculating global mean activation from feature map for image %d" % i)
            
            
            
            for j in range(pool.shape[3]):
        
                temp = pool[i,:,:,j]
                
                mean = temp.mean()
                means.append(mean)
                
            self.mean_activations.append(means)
        
            plt.figure()
            plt.plot(np.arange(128), means)
            plt.title('Feature vector to be hashed for image %d' % i)
            plt.xlabel('Feature Index')
            plt.ylabel('Activation Value')
            plt.show()
        
        # Visualizing every channel in 1st layer activation:
        #for i in range(temp.shape[3]):
            #try plotting the every channel from the first layer
            
        #    plt.matshow(temp[0, :, :, i], cmap='viridis')
        #    plt.axis('off')
        #    plt.show()
        
        ## Visualizing every channel in every intermediate acivation
        # Materials from: https://github.com/gabrielpierobon/cnnshapes/blob/master/README.md
        
        #images_per_row = 16
        
        # Displays the feature maps
        # n_features: Number of features in the feature map
        # size: The feature map has shape (1, size, size, n_features)
        # n_cols: Tiles the activation channels in this matrix
        
        #for layer_name, layer_activation in zip(layer_names, activations):
        #    n_features = layer_activation.shape[-1]
        #    size = layer_activation.shape[1] 
        #    n_cols = n_features // images_per_row 
        #    display_grid = np.zeros((size * n_cols, images_per_row * size))
            
            # Tiles each filter into a big horizontal grid
        #    for col in range(n_cols): 
        #        for row in range(images_per_row):
        #            channel_image = layer_activation[0, :, :, col * images_per_row + row]
                    
                    # Post-processes the feature to make it visually palatable
        #            channel_image -= channel_image.mean()
        #            channel_image /= channel_image.std()
        #            channel_image *= 64
        #            channel_image += 128
        #            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                    
        #            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
                    
        #    scale = 1. / size
        #    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        #    plt.title(layer_name)
        #    plt.grid(False)
        #    plt.imshow(display_grid, aspect='auto', cmap='viridis')
        
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
        
        # visualize hashes    
        # for i in range(len(hashes)):
        
        #    plt.figure()
        #    plt.plot(np.arange(128), hashes[i])
        #    plt.title('Hash for image %d' % i)
        #    plt.xlabel('Feature Index')
        #    plt.ylabel('Hash')
        #    plt.show()
    
    
    def query_image(self, image):
        """ image_hash = hashing_func(processing_func(image))
            find most similar images by hamming distance or some other metric """
            
        image_orig = np.array(image)
        image_orig = cv2.resize(image_orig, (128,128))
        image = image_orig / 255.0
        #print(image.shape)

        matches = []
        hammings = []
        
        # Returns a list of five Numpy arrays: one array per layer activation
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)
        
        activations = self.activation_model.predict(image)
        pool = np.array(activations[-9])
        #print(pool.shape)
        
        for i in range(pool.shape[0]):
            
            image_activations = []
            means = []
            
            #print("Calculating global mean activation from feature map for query image")
            
            for j in range(pool.shape[3]):
        
                temp = pool[i,:,:,j]
                
                mean = temp.mean()
                means.append(mean)
                
            image_activations.append(means)
        
            plt.figure()
            plt.plot(np.arange(128), means)
            plt.title('Feature vector to be hashed for image %d' % i)
            plt.xlabel('Feature Index')
            plt.ylabel('Activation Value')
            plt.show()
        
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
            #def hammingDistance(hash_list_1, hash_list_2):
            #    temp = hash_list_1 ^ hash_list_2
            #    diff = 0
            #
            #    while temp > 0:
            #        diff += temp & 1
            #        temp >>= 1
            #        
            #    return diff
            
            # or using lists:
            # [bit for bit in hash_list_1+hash_list_2 if (bit not in hash_list_1) or (bit not in hash_list_2)]
            
            # https://stackoverflow.com/questions/16312730/comparing-two-lists-and-only-printing-the-differences-xoring-two-lists
            #    return len(set(hash_list_1).symmetric_difference(hash_list_2))
        
        for i in range(len(hammings)):
            if hammings[i] <= 2:
                #print("[Match]: %s" % self.paths[i])
                matches.append(self.paths[i])
                
                im = np.array(self.images[i])
                
                plt.figure()
                plt.imshow(im, cmap="gray")
                plt.axis('off')
                plt.show()
                
                mi = mutual_info_score(image_orig.flatten(), im.astype(dtype=np.uint8).flatten())
                jac = jaccard_similarity_score(image_orig.flatten(), im.astype(dtype=np.uint8).flatten())
                #print("Mutual information: %.3f \t Jaccard score: %.3f" % (mi, jac))

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

def process(img):
    return img

image = Image.open(r"C:\workspace\223C\PNG-v2-mask-applied\A01PR4010.png")
image = np.array(image)

test_dir = r"C:\workspace\223C\PNG-v2-mask-applied"

test = ImageSimilarity2(test_dir, process)
test.query_image(image)
